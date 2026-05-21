"""
manager_app.py — SAM3UI Local Control Panel
─────────────────────────────────────────────
A Gradio app that runs on YOUR LOCAL machine (port 7861) and controls
a SAM3UI instance on VAST.AI.

  - Detects COLD / STANDBY / STARTING / READY state in real-time
  - Provisions a new instance from scratch (cold start)
  - Restarts a paused standby instance (fast restart)
  - Stops the running instance (preserves disk → STANDBY)
  - Force-destroys the instance (COLD — must provision again)
  - Auto-shuts down after 5 minutes of inactivity (configurable)
  - Embeds the remote Gradio app in an iframe when READY
  - Persists your API key and settings in ~/.sam3ui_config.json

Usage:
    python manager_app.py

Then open http://localhost:7861 in your browser.
"""

from __future__ import annotations

import atexit
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

import gradio as gr

from vast_manager import (
    DEFAULT_DISK_GB,
    DEFAULT_GPU_QUERY,
    DEFAULT_IMAGE,
    SHUTDOWN_MINUTES,
    VastManager,
)

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Config persistence ───────────────────────────────────────────────────────

CONFIG_FILE = Path.home() / ".sam3ui_config.json"


def load_config() -> dict:
    if CONFIG_FILE.is_file():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_config(**kwargs) -> None:
    cfg = load_config()
    cfg.update(kwargs)
    try:
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not save config: %s", exc)


# ── Manager singleton ────────────────────────────────────────────────────────
# Shared across all Gradio sessions (single-user tool).

_manager: Optional[VastManager] = None
_manager_lock = threading.Lock()


def get_manager(api_key: str) -> Optional[VastManager]:
    """Return (and lazily create) the VastManager for the given API key."""
    global _manager
    key = api_key.strip()
    if not key:
        return None
    with _manager_lock:
        if _manager is None or _manager.api_key != key:
            if _manager is not None:
                _manager.shutdown()
            _manager = VastManager(api_key=key)
    return _manager


@atexit.register
def _cleanup():
    if _manager is not None:
        _manager.shutdown()


# ── HTML rendering helpers ───────────────────────────────────────────────────

# Visual style for each state
_STATE_STYLE = {
    "cold":     {"label": "COLD",        "color": "#6b7280", "desc": "No instance — GPU not running"},
    "standby":  {"label": "STANDBY",     "color": "#f59e0b", "desc": "Paused — disk preserved, fast restart available"},
    "starting": {"label": "STARTING…",   "color": "#3b82f6", "desc": "Booting — SAM3 not ready yet"},
    "ready":    {"label": "READY  ✓",    "color": "#10b981", "desc": "SAM3 is live — Gradio running"},
    "stopping": {"label": "STOPPING…",   "color": "#f97316", "desc": "Instance stopping"},
    "error":    {"label": "ERROR",        "color": "#ef4444", "desc": "Check logs"},
}


def _fmt_countdown(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


def build_status_html(
    state: str,
    inst: Optional[dict],
    countdown: int,
    url: Optional[str],
) -> str:
    style = _STATE_STYLE.get(state, _STATE_STYLE["error"])
    color = style["color"]
    label = style["label"]
    desc  = style["desc"]

    # Pulse animation for active states
    pulse = "animation:pulse 1.5s infinite;" if state in ("starting", "stopping") else ""

    html = f"""
<style>
  @keyframes pulse {{
    0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }}
  }}
</style>
<div style="font-family:monospace;padding:4px 0;">

  <!-- State badge row -->
  <div style="display:flex;align-items:center;gap:12px;
              padding:14px 18px;background:#1f2937;border-radius:10px;">
    <div style="width:14px;height:14px;border-radius:50%;flex-shrink:0;
                background:{color};box-shadow:0 0 10px {color};{pulse}"></div>
    <div>
      <span style="color:{color};font-size:1.15rem;font-weight:800;
                   letter-spacing:0.05em;">{label}</span>
      <span style="color:#9ca3af;font-size:0.85rem;margin-left:10px;">{desc}</span>
    </div>
  </div>
"""

    # Instance detail row
    if inst:
        iid     = inst.get("id", "?")
        gpu     = f"{inst.get('num_gpus', 1)}× {inst.get('gpu_name', '?')}"
        cost    = f"${inst.get('dph_total', 0.0):.3f}/hr"
        status  = inst.get("status", "?")
        url_tag = (
            f' &nbsp;·&nbsp; <span style="color:#6b7280;">URL:</span> '
            f'<a href="{url}" target="_blank" style="color:#60a5fa;">{url}</a>'
        ) if url else ""

        html += f"""
  <div style="margin-top:6px;padding:10px 18px;background:#111827;
              border-radius:8px;font-size:0.82rem;color:#d1d5db;">
    <span style="color:#6b7280;">ID:</span> {iid} &nbsp;·&nbsp;
    <span style="color:#6b7280;">GPU:</span> {gpu} &nbsp;·&nbsp;
    <span style="color:#6b7280;">Cost:</span> {cost} &nbsp;·&nbsp;
    <span style="color:#6b7280;">Status:</span> {status}{url_tag}
  </div>
"""

    # Countdown bar (only when READY)
    if state == "ready":
        total = SHUTDOWN_MINUTES * 60
        pct   = (countdown / total) * 100 if total > 0 else 0
        bar_color = (
            "#10b981" if pct > 50 else
            "#f59e0b" if pct > 20 else
            "#ef4444"
        )
        html += f"""
  <div style="margin-top:6px;padding:10px 18px;background:#111827;border-radius:8px;">
    <div style="display:flex;justify-content:space-between;
                font-size:0.8rem;color:#9ca3af;margin-bottom:5px;">
      <span>Auto-shutdown in</span>
      <span style="color:{bar_color};font-weight:700;font-size:0.95rem;">
        {_fmt_countdown(countdown)}
      </span>
    </div>
    <div style="height:5px;background:#374151;border-radius:3px;overflow:hidden;">
      <div style="width:{pct:.1f}%;height:100%;background:{bar_color};
                  border-radius:3px;transition:width 2s linear;"></div>
    </div>
  </div>
"""

    html += "</div>"
    return html


def build_iframe_html(url: Optional[str], state: str) -> str:
    if state == "cold":
        icon, msg, sub = "🧊", "Instance is COLD", "Click Start (Cold) to provision a GPU"
    elif state == "standby":
        icon, msg, sub = "💤", "Instance is on STANDBY", "Click Restart to wake it up (~1-2 min)"
    elif state == "starting":
        icon, msg, sub = "⚙️", "Instance is starting…", "SAM3 will appear here when ready (check progress above)"
    elif state == "stopping":
        icon, msg, sub = "⏸", "Instance is stopping…", "It will enter STANDBY shortly"
    else:
        icon, msg, sub = "❓", "Unknown state", ""

    if not url or state != "ready":
        return f"""
<div style="display:flex;align-items:center;justify-content:center;
            height:620px;background:#0f172a;border-radius:12px;
            border:1px solid #1e293b;font-family:sans-serif;">
  <div style="text-align:center;color:#64748b;">
    <div style="font-size:3.5rem;margin-bottom:12px;">{icon}</div>
    <div style="font-size:1.1rem;font-weight:600;color:#94a3b8;">{msg}</div>
    <div style="font-size:0.85rem;margin-top:6px;">{sub}</div>
  </div>
</div>"""

    return f"""
<div style="border-radius:12px;overflow:hidden;border:1px solid #1e293b;
            box-shadow:0 4px 24px rgba(0,0,0,0.4);">
  <div style="background:#0f172a;padding:8px 14px;display:flex;
              align-items:center;gap:8px;font-family:monospace;font-size:0.8rem;color:#64748b;">
    <span style="color:#10b981;font-size:0.7rem;">●</span>
    <a href="{url}" target="_blank" style="color:#60a5fa;text-decoration:none;">{url}</a>
    <span style="margin-left:auto;">
      <a href="{url}" target="_blank"
         style="color:#94a3b8;text-decoration:none;font-size:0.78rem;">
        ↗ Open in new tab
      </a>
    </span>
  </div>
  <iframe src="{url}" width="100%" height="680px" frameborder="0"
          style="display:block;" allowfullscreen></iframe>
</div>"""


# ── Gradio event handlers ────────────────────────────────────────────────────

def _status_update(api_key: str):
    """Common refresh — returns (status_html, iframe_html, log_text)."""
    mgr = get_manager(api_key)
    if not mgr:
        return (
            build_status_html("cold", None, 0, None),
            build_iframe_html(None, "cold"),
            "(no instance)",
        )
    state, inst = mgr.get_state()
    url         = mgr.get_gradio_url(inst) if inst else None
    countdown   = mgr.countdown_seconds()
    logs        = mgr.get_logs(inst) if inst else "(no instance running)"
    return (
        build_status_html(state, inst, countdown, url),
        build_iframe_html(url, state),
        logs,
    )


def on_timer_tick(api_key):
    return _status_update(api_key)


def on_start_cold(api_key, hf_token, gpu_query, disk_gb, docker_image):
    """Provision a brand-new instance (cold start)."""
    if not api_key.strip():
        return "⚠ Enter your VAST API key first."
    if not hf_token.strip():
        return "⚠ HuggingFace token is required to download the SAM3 checkpoint."

    mgr   = get_manager(api_key)
    state, _ = mgr.get_state()
    if state in ("starting", "ready"):
        mgr.reset_activity()
        return "Instance is already running."
    if state == "standby":
        return "Instance is on STANDBY — use Restart instead (much faster)."

    save_config(
        api_key=api_key, hf_token=hf_token,
        gpu_query=gpu_query, disk_gb=int(disk_gb), docker_image=docker_image,
    )
    return mgr.provision(
        gpu_query=gpu_query,
        image=docker_image,
        disk_gb=int(disk_gb),
        hf_token=hf_token,
    )


def on_restart(api_key, *_):
    """Restart a STANDBY instance (fast path — no reinstall)."""
    mgr = get_manager(api_key)
    if not mgr:
        return "⚠ Enter your VAST API key first."
    mgr.reset_activity()
    return mgr.start()


def on_stop(api_key, *_):
    """Stop the instance, preserving the disk (→ STANDBY)."""
    mgr = get_manager(api_key)
    if not mgr:
        return "⚠ Enter your VAST API key first."
    return mgr.stop()


def on_destroy(api_key, *_):
    """Permanently destroy the instance (→ COLD)."""
    mgr = get_manager(api_key)
    if not mgr:
        return "⚠ Enter your VAST API key first."
    return mgr.destroy()


def on_reset_timer(api_key, *_):
    """Reset the 5-minute inactivity countdown."""
    mgr = get_manager(api_key)
    if not mgr:
        return "⚠ Enter your VAST API key first."
    mgr.reset_activity()
    return f"Timer reset — auto-shutdown in {SHUTDOWN_MINUTES}:00."


def on_api_key_change(api_key):
    """Immediately refresh status when the API key is entered."""
    return _status_update(api_key)

# ── Gradio UI ────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    cfg = load_config()

    with gr.Blocks(title="SAM3UI — Instance Manager") as demo:

        # ── Header ───────────────────────────────────────────────────────────
        gr.Markdown(
            "# SAM3UI — VAST.AI Instance Manager\n"
            "Control your GPU instance lifecycle and access SAM3 from one place."
        )

        # ── Configuration ────────────────────────────────────────────────────
        with gr.Accordion("⚙ Configuration", open=not bool(cfg.get("api_key"))):
            gr.Markdown(
                "Get your VAST API key from "
                "[console.vast.ai/manage-keys](https://console.vast.ai/manage-keys) &nbsp;·&nbsp; "
                "Get your HuggingFace token from "
                "[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)"
            )
            with gr.Row():
                api_key = gr.Textbox(
                    label="VAST API Key",
                    type="password",
                    value=cfg.get("api_key", ""),
                    placeholder="Paste your VAST API key here",
                    scale=1,
                )
                hf_token = gr.Textbox(
                    label="HuggingFace Token",
                    type="password",
                    value=cfg.get("hf_token", ""),
                    placeholder="hf_xxxxxxxxxxxxxxxx",
                    scale=1,
                )
            with gr.Row():
                gpu_query = gr.Textbox(
                    label="GPU Offer Query",
                    value=cfg.get("gpu_query", DEFAULT_GPU_QUERY),
                    info="VAST search query — gpu_ram>=24 means 24 GB+ VRAM (3090 class and above); dph_total<0.50 caps cost at $0.50/hr",
                    scale=3,
                )
                disk_gb = gr.Number(
                    label="Disk (GB)",
                    value=cfg.get("disk_gb", DEFAULT_DISK_GB),
                    precision=0,
                    minimum=20,
                    maximum=500,
                    scale=1,
                )
            docker_image = gr.Textbox(
                label="Docker Image",
                value=cfg.get("docker_image", DEFAULT_IMAGE),
                info="Base image used when provisioning (must have CUDA)",
            )

        # ── Status panel ─────────────────────────────────────────────────────
        status_html = gr.HTML(
            value=build_status_html("cold", None, 0, None),
            elem_classes=["status-panel"],
        )

        # ── Controls ─────────────────────────────────────────────────────────
        with gr.Row(elem_classes=["controls-row"]):
            start_cold_btn = gr.Button(
                "▶  Start (Cold)",
                variant="primary",
                size="lg",
            )
            restart_btn = gr.Button(
                "↺  Restart",
                variant="secondary",
                size="lg",
            )
            stop_btn = gr.Button(
                "⏸  Stop",
                variant="secondary",
                size="lg",
            )
            reset_timer_btn = gr.Button(
                "⏱  Reset Timer",
                variant="secondary",
                size="lg",
            )
            destroy_btn = gr.Button(
                "💥  Force Destroy",
                variant="stop",
                size="lg",
            )

        action_log = gr.Textbox(
            label="Last Action",
            interactive=False,
            max_lines=2,
            placeholder="Action results will appear here…",
        )

        gr.Markdown(
            "---\n"
            "### SAM3 Interface\n"
            "_The remote Gradio app loads here when the instance is READY. "
            "You can also click the URL above to open it in a new tab._"
        )

        # ── Embedded SAM3 UI ─────────────────────────────────────────────────
        iframe_html = gr.HTML(
            value=build_iframe_html(None, "cold"),
        )

        # ── Live instance log ─────────────────────────────────────────────────
        with gr.Accordion("📋 Instance Log (last 100 lines)", open=True):
            instance_log = gr.Textbox(
                label=None,
                value="(no instance running)",
                interactive=False,
                lines=18,
                max_lines=18,
                elem_id="instance-log",
            )

        # ── Timer (polls every 5 s) ──────────────────────────────────────────
        timer = gr.Timer(value=5)

        # ── Event wiring ──────────────────────────────────────────────────────
        all_cfg_inputs = [api_key, hf_token, gpu_query, disk_gb, docker_image]

        # Refresh status + iframe + log on every timer tick
        timer.tick(
            fn=on_timer_tick,
            inputs=[api_key],
            outputs=[status_html, iframe_html, instance_log],
        )

        # Also refresh immediately when the API key is entered
        api_key.change(
            fn=on_api_key_change,
            inputs=[api_key],
            outputs=[status_html, iframe_html, instance_log],
        )

        # Button actions — each returns a log message, then status refreshes
        def _wrap(action_fn):
            """Return a function that calls action_fn then refreshes status."""
            def _inner(*args):
                msg = action_fn(*args)
                status, iframe, logs = _status_update(args[0])  # args[0] = api_key
                return msg, status, iframe, logs
            return _inner

        for btn, fn in [
            (start_cold_btn,  on_start_cold),
            (restart_btn,     on_restart),
            (stop_btn,        on_stop),
            (destroy_btn,     on_destroy),
            (reset_timer_btn, on_reset_timer),
        ]:
            btn.click(
                fn=_wrap(fn),
                inputs=all_cfg_inputs,
                outputs=[action_log, status_html, iframe_html, instance_log],
            )

    return demo


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,       # 7860 is reserved for the remote app running on VAST
        theme=gr.themes.Soft(),
        css="""
        .status-panel { margin-bottom: 0 !important; }
        .controls-row button { min-width: 130px; }
        """,
        share=False,
    )
