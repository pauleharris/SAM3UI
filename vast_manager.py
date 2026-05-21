"""
vast_manager.py — VAST.AI instance lifecycle manager for SAM3UI
────────────────────────────────────────────────────────────────
Manages the full state machine for a labeled VAST.AI instance:

  COLD       No instance exists. Provisioning required (5-15 min cold start).
  STANDBY    Instance exists but is stopped. Disk preserved; fast restart (~1-2 min).
  STARTING   Instance is running but Gradio is not yet responding.
  READY      Gradio is accessible — SAM3 is live.
  STOPPING   Instance is being stopped.

Auto-shutdown:
  A background daemon thread monitors the inactivity timer. When the countdown
  reaches zero while the state is READY, the instance is stopped (not destroyed),
  leaving it in STANDBY for a fast restart next time.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

INSTANCE_LABEL   = "sam3ui"
GRADIO_PORT      = 7860
REPO_URL         = "https://github.com/pauleharris/SAM3UI.git"
DEFAULT_IMAGE    = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DEFAULT_GPU_QUERY = "gpu_ram>=24 dph_total<0.50 reliability>0.99 num_gpus=1 inet_up>300 cuda_vers>=12.0 direct_port_count>1"
DEFAULT_DISK_GB  = 30
SHUTDOWN_MINUTES = 5

# ── State definitions ─────────────────────────────────────────────────────────

STATES = ("cold", "standby", "starting", "ready", "stopping", "error")

# VAST status strings that map to each of our logical states
_VAST_STATUS_MAP = {
    "cold":     {"destroyed", "delerr", "failed"},
    "standby":  {"stopped", "exited"},
    "starting": {"loading", "running", "provisioning"},  # running but Gradio not up
    "stopping": {"stopping"},
}


class VastManager:
    """
    Manages a single labeled VAST.AI instance through its lifecycle.

    Thread-safe: all public methods can be called from Gradio event handlers
    or the background monitor simultaneously.
    """

    def __init__(
        self,
        api_key: str,
        shutdown_minutes: int = SHUTDOWN_MINUTES,
    ) -> None:
        try:
            from vastai import VastAI
        except ImportError:
            raise ImportError(
                "vastai is not installed.  Run:  pip install vastai"
            )

        self.api_key          = api_key.strip()
        self.shutdown_seconds = shutdown_minutes * 60
        self._vast            = VastAI(api_key=self.api_key)
        self._lock            = threading.Lock()
        self._last_activity   = time.time()
        self._stop_event      = threading.Event()
        self._monitor_thread  = threading.Thread(
            target=self._monitor_loop, daemon=True, name="sam3-shutdown-monitor"
        )
        self._monitor_thread.start()
        logger.info("VastManager initialised (shutdown after %d min inactivity)", shutdown_minutes)

    # ── Instance discovery ───────────────────────────────────────────────────

    def find_instance(self) -> Optional[dict]:
        """Return the dict for our labeled instance, or None."""
        try:
            instances = self._vast.show_instances() or []
            for inst in instances:
                if inst.get("label") == INSTANCE_LABEL:
                    return inst
        except Exception as exc:
            logger.error("find_instance error: %s", exc)
        return None

    def get_state(self) -> tuple[str, Optional[dict]]:
        """
        Poll VAST and return (state, instance_dict).

        State is one of: cold | standby | starting | ready | stopping | error
        """
        try:
            inst = self.find_instance()
        except Exception as exc:
            logger.error("get_state error: %s", exc)
            return "error", None

        if inst is None:
            return "cold", None

        vast_status = inst.get("status", "").lower()

        if vast_status in _VAST_STATUS_MAP["cold"]:
            return "cold", None  # treat as if no instance

        if vast_status in _VAST_STATUS_MAP["standby"]:
            return "standby", inst

        if vast_status in _VAST_STATUS_MAP["stopping"]:
            return "stopping", inst

        # Instance reports "running" — check whether Gradio is actually up
        if vast_status == "running":
            url = self.get_gradio_url(inst)
            if url and self._check_url(url):
                return "ready", inst
            return "starting", inst

        # loading / provisioning / anything else → still starting
        return "starting", inst

    # ── URL helpers ──────────────────────────────────────────────────────────

    def get_gradio_url(self, inst: dict) -> Optional[str]:
        """Extract the public Gradio URL from the VAST port mapping."""
        ports = inst.get("ports") or {}
        mappings = ports.get(f"{GRADIO_PORT}/tcp") or []
        if mappings:
            host_port = mappings[0].get("HostPort")
            public_ip = inst.get("public_ipaddr", "").strip()
            if host_port and public_ip:
                return f"http://{public_ip}:{host_port}"
        return None

    def _check_url(self, url: str, timeout: float = 3.0) -> bool:
        """Return True if the URL returns HTTP 200."""
        try:
            r = httpx.get(url, timeout=timeout, follow_redirects=True)
            return r.status_code == 200
        except Exception:
            return False

    def get_logs(self, inst: dict, tail: int = 100) -> str:
        """Fetch the last *tail* lines of container logs for *inst*."""
        try:
            return self._vast.logs(instance_id=inst["id"], tail=str(tail)) or "(no output yet)"
        except Exception as exc:
            return f"(log fetch error: {exc})"

    # ── Lifecycle actions ────────────────────────────────────────────────────

    def provision(
        self,
        gpu_query:  str = DEFAULT_GPU_QUERY,
        image:      str = DEFAULT_IMAGE,
        disk_gb:    int = DEFAULT_DISK_GB,
        hf_token:   str = "",
    ) -> str:
        """
        Find the cheapest offer matching *gpu_query* and create a new instance.
        The on-start script installs SAM3 and launches app.py.
        """
        try:
            offers = self._vast.search_offers(query=gpu_query)
        except Exception as exc:
            return f"Error searching offers: {exc}"

        if not offers:
            return f"No offers found for query: '{gpu_query}'. Try broadening the search."

        cheapest = min(offers, key=lambda o: o.get("dph_total", 9999.0))
        offer_id = cheapest["id"]
        cost     = cheapest.get("dph_total", 0.0)
        gpu      = cheapest.get("gpu_name", "?")

        logger.info("Provisioning offer %s  %s  $%.3f/hr", offer_id, gpu, cost)

        onstart = self._build_onstart(hf_token)
        try:
            self._vast.create_instance(
                id=offer_id,
                image=image,
                disk=disk_gb,
                onstart_cmd=onstart,
                label=INSTANCE_LABEL,
            )
        except Exception as exc:
            return f"Error creating instance: {exc}"

        self.reset_activity()
        return (
            f"Instance provisioning started — {gpu}, ${cost:.3f}/hr. "
            f"Expect 5-15 min for first-time setup."
        )

    def start(self) -> str:
        """Restart a STANDBY instance (fast — skips re-installation)."""
        inst = self.find_instance()
        if inst is None:
            return "No instance found. Use Start (Cold) to provision one."

        status = inst.get("status", "").lower()
        if status in ("stopped", "exited"):
            try:
                self._vast.start_instance(id=inst["id"])
                self.reset_activity()
                return "Restarting from standby (~1-2 min until ready)."
            except Exception as exc:
                return f"Error restarting: {exc}"

        if status == "running":
            return "Instance is already running."

        return f"Instance is in state '{status}' — cannot restart."

    def stop(self) -> str:
        """
        Stop the running instance.
        Disk is preserved so it can be restarted quickly (STANDBY state).
        """
        inst = self.find_instance()
        if inst is None:
            return "No running instance found."
        try:
            self._vast.stop_instance(id=inst["id"])
            return "Stopping… disk preserved for fast restart."
        except Exception as exc:
            return f"Error stopping: {exc}"

    def destroy(self) -> str:
        """Permanently destroy the instance and its disk."""
        inst = self.find_instance()
        if inst is None:
            return "No instance found."
        try:
            self._vast.destroy_instance(id=inst["id"])
            return "Instance destroyed (COLD — full provisioning needed next time)."
        except Exception as exc:
            return f"Error destroying: {exc}"

    # ── Activity / countdown ─────────────────────────────────────────────────

    def reset_activity(self) -> None:
        """Reset the inactivity timer (call on every user interaction)."""
        with self._lock:
            self._last_activity = time.time()

    def countdown_seconds(self) -> int:
        """Seconds remaining before auto-shutdown triggers."""
        with self._lock:
            elapsed = time.time() - self._last_activity
            return max(0, int(self.shutdown_seconds - elapsed))

    # ── Background auto-shutdown monitor ─────────────────────────────────────

    def _monitor_loop(self) -> None:
        """
        Daemon thread: checks every 30 s.
        When countdown reaches zero and the instance is READY, stops it.
        """
        while not self._stop_event.wait(30):
            if self.countdown_seconds() > 0:
                continue
            try:
                state, inst = self.get_state()
                if state == "ready" and inst:
                    logger.info("Auto-shutdown: stopping instance %s due to inactivity.", inst["id"])
                    self._vast.stop_instance(id=inst["id"])
                    # Back-date last_activity so the next check doesn't re-trigger immediately
                    with self._lock:
                        self._last_activity = time.time() - self.shutdown_seconds + 60
            except Exception as exc:
                logger.error("Auto-shutdown monitor error: %s", exc)

    def shutdown(self) -> None:
        """Stop the background monitor thread (call when app exits)."""
        self._stop_event.set()

    # ── On-start script template ─────────────────────────────────────────────

    @staticmethod
    def _build_onstart(hf_token: str) -> str:
        """
        Build the on-start bash script that runs on the VAST instance.

        Idempotent: skips installation if /workspace/SAM3UI already exists,
        so that restarting a STANDBY instance is fast (~30 s) vs cold start (~15 min).
        """
        return f"""#!/bin/bash
# SAM3UI on-start script — runs once when the container starts.
# Logs everything to /workspace/sam3ui_setup.log

LOG=/workspace/sam3ui_setup.log
exec > >(tee -a "$LOG") 2>&1

log() {{ echo "[$(date '+%H:%M:%S')] $*"; }}

log "=== SAM3UI on-start ==="

# ── First-time installation (skipped on warm restart) ─────────────────────
if [ ! -f /workspace/SAM3UI/.installed ]; then
    log "Cold start — installing dependencies (this takes ~10-15 min)…"

    apt-get update -qq
    apt-get install -y -qq git curl

    cd /workspace
    git clone {REPO_URL} SAM3UI || {{ log "ERROR: git clone failed"; exit 1; }}
    cd SAM3UI

    # torch already present in the devel image — just install app deps
    pip install -r requirements.txt -q

    # SAM3 from source
    pip install git+https://github.com/facebookresearch/sam3.git -q \
        || log "WARNING: sam3 pip install failed — app will show an error but will still start"

    # Download model checkpoint
    export HF_TOKEN="{hf_token}"
    export HUGGING_FACE_HUB_TOKEN="{hf_token}"
    mkdir -p checkpoints
    # Use 'hf' CLI (huggingface-cli is deprecated in newer hf_hub versions)
    hf download facebook/sam3-large --local-dir checkpoints/ \
        || log "WARNING: HF download failed — check your token and model access"

    touch /workspace/SAM3UI/.installed
    log "Installation complete."
else
    log "Warm restart — skipping installation."
fi

# ── Launch the Gradio app ──────────────────────────────────────────────────
cd /workspace/SAM3UI
APP_LOG=/workspace/sam3ui.log
log "Launching app.py — output in $APP_LOG"
nohup python app.py > "$APP_LOG" 2>&1 &
APP_PID=$!
log "app.py started (PID $APP_PID). Setup complete."
"""
