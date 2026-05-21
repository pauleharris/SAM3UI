"""
vast_manager.py — VAST.AI instance lifecycle manager for SAM3UI
────────────────────────────────────────────────────────────────
State machine:

  COLD       No instance exists (or in terminal failure state).
  STANDBY    Instance exists but container is stopped. Disk preserved.
  STARTING   Container is running but Gradio not yet responding.
  READY      Gradio is accessible — SAM is live.
  STOPPING   Container is tearing down.
  UNKNOWN    VAST returned a status string we have not seen before.
  ERROR      Could not reach the VAST API.

Auto-shutdown:
  A background daemon stops the instance after SHUTDOWN_MINUTES of inactivity,
  leaving it in STANDBY for a fast restart.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

INSTANCE_LABEL    = "sam3ui"
GRADIO_PORT       = 7860
REPO_URL          = "https://github.com/pauleharris/SAM3UI.git"
DEFAULT_IMAGE     = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DEFAULT_GPU_QUERY = (
    "gpu_ram>=24 dph_total<0.50 reliability>0.995 num_gpus=1 "
    "inet_up>500 cuda_vers>=12.0 direct_port_count>2"
)
DEFAULT_DISK_GB   = 30
SHUTDOWN_MINUTES  = 5

# ── Comprehensive VAST status → logical state map ─────────────────────────────
# Any status NOT in this dict returns "unknown" — never silently "starting".

_STATUS_TO_STATE: dict[str, str] = {
    # ── No status yet — container still being allocated by VAST ──────────
    "":              "starting",
    # ── Terminal / gone ───────────────────────────────────────────────────
    "destroyed":    "cold",
    "delerr":       "cold",
    "delunable":    "cold",
    "failed":       "cold",
    "error":        "cold",
    # ── Container exists but stopped ──────────────────────────────────────
    "stopped":      "standby",
    "exited":       "standby",
    "hibernated":   "standby",
    "hibernate":    "standby",
    "paused":       "standby",
    "suspended":    "standby",
    # ── Container coming up ───────────────────────────────────────────────
    "provisioning": "starting",
    "loading":      "starting",
    "starting":     "starting",
    "created":      "starting",
    # ── Container tearing down ────────────────────────────────────────────
    "stopping":     "stopping",
    "destroying":   "stopping",
    # ── Container running — must verify Gradio ────────────────────────────
    "running":      "_check_gradio",
}

STATES = ("cold", "standby", "starting", "ready", "stopping", "unknown", "error")


def _get_status(inst: dict) -> str:
    """
    Extract the current status string from a VAST instance dict.

    The VAST REST API uses 'actual_status' / 'cur_state'; older SDK versions
    exposed it as 'status'.  Try all three and return lowercase or "".
    """
    raw = (
        inst.get("actual_status") or
        inst.get("cur_state") or
        inst.get("status") or
        ""
    )
    return raw.lower().strip()


class VastManager:
    """
    Manages a single labeled VAST.AI instance through its lifecycle.
    Thread-safe: all public methods can be called from Gradio event handlers
    or the background monitor simultaneously.
    """

    def __init__(self, api_key: str, shutdown_minutes: int = SHUTDOWN_MINUTES) -> None:
        try:
            from vastai import VastAI  # type: ignore[import]
        except ImportError:
            raise ImportError("vastai is not installed.  Run:  pip install vastai")

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

    # ── Raw API helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_instances(raw) -> list[dict]:
        """
        Normalize whatever show_instances() returns into a plain list of dicts.

        The vastai SDK has returned different shapes across versions:
          list of dicts | dict with "instances" key | iterator | None
        """
        if raw is None:
            return []
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
        if isinstance(raw, dict):
            for key in ("instances", "results", "data"):
                if key in raw and isinstance(raw[key], list):
                    return [x for x in raw[key] if isinstance(x, dict)]
            return []
        try:
            return [x for x in raw if isinstance(x, dict)]
        except Exception:
            return []

    def show_all_instances(self) -> list[dict]:
        """Return every VAST instance on this account. Raises on API error."""
        raw = self._vast.show_instances()
        return self._parse_instances(raw)

    # ── Instance discovery ────────────────────────────────────────────────────

    def find_instance(self) -> Optional[dict]:
        """Return the dict for our labeled instance, or None if not found."""
        try:
            for inst in self.show_all_instances():
                if inst.get("label") == INSTANCE_LABEL:
                    return inst
        except Exception as exc:
            logger.error("find_instance error: %s", exc)
        return None

    def get_state(self) -> tuple[str, Optional[dict]]:
        """
        Poll VAST and return (state, instance_dict).

        state is one of: cold | standby | starting | ready | stopping | unknown | error
        instance_dict is None when state is cold or error.
        """
        try:
            inst = self.find_instance()
        except Exception as exc:
            logger.error("get_state error: %s", exc)
            return "error", None

        if inst is None:
            return "cold", None

        vast_status = _get_status(inst)
        mapped = _STATUS_TO_STATE.get(vast_status)

        if mapped is None:
            logger.warning(
                "Unknown VAST status %r for instance %s — showing UNKNOWN",
                vast_status, inst.get("id"),
            )
            return "unknown", inst

        if mapped == "cold":
            return "cold", None   # terminal state — treat as no instance

        if mapped in ("standby", "stopping", "starting"):
            return mapped, inst

        if mapped == "_check_gradio":
            url = self.get_gradio_url(inst)
            if url and self._check_url(url):
                return "ready", inst
            return "starting", inst

        return "unknown", inst  # should not reach here

    # ── URL helpers ───────────────────────────────────────────────────────────

    def get_gradio_url(self, inst: dict) -> Optional[str]:
        """
        Extract the public Gradio URL from the VAST port mapping.
        Tries both "7860/tcp" and "7860" as keys, and multiple casing variants.
        """
        ports     = inst.get("ports") or {}
        public_ip = (inst.get("public_ipaddr") or "").strip()
        if not public_ip:
            return None

        for key in (f"{GRADIO_PORT}/tcp", str(GRADIO_PORT)):
            mappings = ports.get(key)
            if not mappings:
                continue
            entry = mappings[0] if isinstance(mappings, list) else mappings
            if isinstance(entry, dict):
                host_port = (
                    entry.get("HostPort") or entry.get("hostPort") or
                    entry.get("host_port") or entry.get("port") or ""
                )
            else:
                host_port = str(entry)
            if host_port:
                return f"http://{public_ip}:{host_port}"
        return None

    def _check_url(self, url: str, timeout: float = 3.0) -> bool:
        """Return True if the URL responds with HTTP 2xx."""
        try:
            r = httpx.get(url, timeout=timeout, follow_redirects=True)
            return r.status_code < 400
        except Exception:
            return False

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_diagnostics(self, log_tail: int = 60) -> str:
        """
        Return a rich diagnostic string for the Instance Log panel showing:
          1. All VAST instances with their raw status strings
          2. Our instance: state mapping, URL, Gradio reachability, raw ports
          3. Container log (last log_tail lines — only when container is running)
        """
        SEP   = "─" * 56
        lines: list[str] = []

        # ── All instances ─────────────────────────────────────────────────
        lines.append(f"┌ {SEP}")
        lines.append("│  VAST.AI LIVE STATUS")
        lines.append(f"│  {SEP}")

        our_inst: Optional[dict] = None
        api_error: Optional[str] = None

        try:
            all_insts = self.show_all_instances()
            lines.append(f"│  show_instances() → {len(all_insts)} total instance(s)")
            lines.append("│")
            if all_insts:
                for inst in all_insts:
                    iid    = inst.get("id", "?")
                    lbl    = inst.get("label") or "(no label)"
                    status = _get_status(inst) or "?"
                    gpu    = inst.get("gpu_name", "?")
                    cost   = f"${inst.get('dph_total', 0):.3f}/hr"
                    marker = "  ◄ OUR INSTANCE" if lbl == INSTANCE_LABEL else ""
                    lines.append(
                        f"│  #{iid}  [{status}]  {gpu}  {cost}"
                        f"  label={lbl!r}{marker}"
                    )
                    if lbl == INSTANCE_LABEL:
                        our_inst = inst
            else:
                lines.append("│  (no instances on this account)")
        except Exception as exc:
            api_error = str(exc)
            lines.append(f"│  ERROR calling show_instances(): {exc}")

        # ── Our instance detail ───────────────────────────────────────────
        lines.append("│")
        if our_inst:
            vast_status = _get_status(our_inst)
            mapped      = _STATUS_TO_STATE.get(vast_status, "unknown")
            url         = self.get_gradio_url(our_inst)

            if mapped == "_check_gradio":
                gradio_up = bool(url and self._check_url(url))
                logical   = "READY" if gradio_up else "STARTING (waiting for Gradio)"
            elif mapped == "cold":
                gradio_up = False
                logical   = "COLD (terminal — treating as no instance)"
            else:
                gradio_up = False
                logical   = mapped.upper()

            lines.append(f"│  Our instance #{our_inst.get('id')}:")
            raw_actual   = our_inst.get("actual_status")
            raw_cur      = our_inst.get("cur_state")
            raw_intended = our_inst.get("intended_status")
            lines.append(f"│    actual_status : {raw_actual!r}")
            lines.append(f"│    cur_state     : {raw_cur!r}")
            lines.append(f"│    intended      : {raw_intended!r}")
            lines.append(f"│    Logical state : {logical}")
            lines.append(f"│    GPU           : {our_inst.get('num_gpus', 1)}× {our_inst.get('gpu_name', '?')}")
            lines.append(f"│    Cost          : ${our_inst.get('dph_total', 0):.3f}/hr")
            lines.append(f"│    Gradio URL    : {url or '(no port mapping yet)'}")
            if url:
                lines.append(
                    f"│    Gradio alive  : {'YES ✓' if gradio_up else 'NO (not responding yet)'}"
                )
            raw_ports = our_inst.get("ports") or {}
            lines.append(f"│    Raw ports     : {raw_ports if raw_ports else '(empty)'}")
        elif not api_error:
            lines.append(f"│  No instance with label {INSTANCE_LABEL!r} found.")

        lines.append(f"└ {SEP}")
        lines.append("")

        # ── Container log ─────────────────────────────────────────────────
        if our_inst:
            cstatus = _get_status(our_inst)
            if cstatus in ("running", "loading", "provisioning", "starting", "created"):
                lines.append(f"── Container Log (last {log_tail} lines) {'─' * 18}")
                lines.append(self.get_logs(our_inst, tail=log_tail))
            else:
                lines.append(f"── Container Log {'─' * 38}")
                lines.append(
                    f"(container is {cstatus!r}"
                    " — logs only available when running)"
                )
        else:
            lines.append(f"── Container Log {'─' * 38}")
            lines.append("(no instance)")

        return "\n".join(lines)

    def get_logs(self, inst: dict, tail: int = 100) -> str:
        """Fetch the last *tail* lines of container logs, filtering VAST noise."""
        try:
            raw = self._vast.logs(instance_id=inst["id"], tail=str(tail)) or ""
        except Exception as exc:
            return f"(log fetch error: {exc})"

        _NOISE = (
            "remote port forwarding failed",
            "Warning: Permanently added",
            "Server listening on",
            "invoke-rc.d:",
            "debconf:",
            "policy-rc.d denied",
        )
        lines = [ln for ln in raw.splitlines() if not any(n in ln for n in _NOISE)]
        return "\n".join(lines) if lines else "(no output yet)"

    # ── Lifecycle actions ─────────────────────────────────────────────────────

    def provision(
        self,
        gpu_query: str = DEFAULT_GPU_QUERY,
        image:     str = DEFAULT_IMAGE,
        disk_gb:   int = DEFAULT_DISK_GB,
        hf_token:  str = "",
    ) -> str:
        """Find cheapest matching offer and create a new labeled instance."""
        try:
            offers = self._vast.search_offers(query=gpu_query)
        except Exception as exc:
            return f"Error searching offers: {exc}"

        if not offers:
            return f"No offers found for: '{gpu_query}'. Try broadening the search."

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
            f"Provisioning started — {gpu}, ${cost:.3f}/hr. "
            "Expect 10-15 min for first-time setup."
        )

    def start(self) -> str:
        """Restart a STANDBY instance (fast — skips re-installation)."""
        inst = self.find_instance()
        if inst is None:
            return "No instance found. Use Start (Cold) to provision one."

        status = _get_status(inst)
        if _STATUS_TO_STATE.get(status) == "standby":
            try:
                self._vast.start_instance(id=inst["id"])
                self.reset_activity()
                return "Restarting from standby (~1-2 min until ready)."
            except Exception as exc:
                err = str(exc)
                if "no such container" in err.lower():
                    return "Host lost the container — use Force Destroy then Start (Cold)."
                return f"Error restarting: {exc}"

        if status == "running":
            return "Instance is already running."

        return (
            f"Cannot restart — instance is {status!r}. "
            "Use Force Destroy if stuck."
        )

    def stop(self) -> str:
        """Stop the running instance (disk preserved → STANDBY)."""
        inst = self.find_instance()
        if inst is None:
            return "No running instance found."
        try:
            self._vast.stop_instance(id=inst["id"])
            return "Stopping… disk preserved for fast restart."
        except Exception as exc:
            err = str(exc)
            if "no such container" in err.lower():
                return "Host lost the container — use Force Destroy then Start (Cold)."
            return f"Error stopping: {exc}"

    def destroy(self) -> str:
        """Permanently destroy the instance and its disk (→ COLD)."""
        inst = self.find_instance()
        if inst is None:
            return "No instance found — already cold."
        try:
            self._vast.destroy_instance(id=inst["id"])
            return "Instance destroyed (COLD — full provisioning needed next time)."
        except Exception as exc:
            err = str(exc)
            if "no such container" in err.lower():
                return "Instance already gone — COLD (full provisioning needed next time)."
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
        """Daemon thread: auto-stops the instance after inactivity timeout."""
        while not self._stop_event.wait(30):
            if self.countdown_seconds() > 0:
                continue
            try:
                state, inst = self.get_state()
                if state == "ready" and inst:
                    logger.info(
                        "Auto-shutdown: stopping instance %s due to inactivity.",
                        inst["id"],
                    )
                    self._vast.stop_instance(id=inst["id"])
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
    pip install "numpy>=2.0.0" --upgrade -q  # force upgrade before requirements
    pip install -r requirements.txt -q

    # Pre-cache models from HuggingFace Hub so first inference is instant
    export HF_TOKEN="{hf_token}"
    export HUGGING_FACE_HUB_TOKEN="{hf_token}"
    log "Pre-caching Grounding DINO model…"
    python -c "
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')
print('Grounding DINO cached')
" || log "WARNING: Grounding DINO pre-cache failed — will download on first use"
    log "Pre-caching SAM model…"
    python -c "
from transformers import SamModel, SamProcessor
SamProcessor.from_pretrained('facebook/sam-vit-large')
SamModel.from_pretrained('facebook/sam-vit-large')
print('SAM cached')
" || log "WARNING: SAM pre-cache failed — will download on first use"

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
