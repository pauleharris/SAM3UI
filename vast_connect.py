"""
vast_connect.py — Find your running SAM3UI instance on VAST.AI
──────────────────────────────────────────────────────────────
Usage:
    python vast_connect.py

Reads your VAST API key from (in order):
  1. VAST_API_KEY environment variable
  2. ~/.vast_api_key  (written by `vastai set api-key YOUR_KEY`)

Prints the Gradio URL for every running instance that has port 7860 mapped.

Prerequisites:
    pip install vastai
    vastai set api-key YOUR_KEY     # one-time setup
      -- OR --
    export VAST_API_KEY=YOUR_KEY
"""

from __future__ import annotations

import os
import sys

try:
    from vastai import VastAI
except ImportError:
    print("ERROR: vastai is not installed.  Run:  pip install vastai")
    sys.exit(1)


GRADIO_PORT = 7860


def get_api_key() -> str | None:
    """Return the VAST API key from env var or ~/.vast_api_key file."""
    key = os.environ.get("VAST_API_KEY", "").strip()
    if key:
        return key

    key_file = os.path.expanduser("~/.vast_api_key")
    if os.path.isfile(key_file):
        with open(key_file) as f:
            return f.read().strip()

    return None


def extract_host_port(instance: dict, container_port: int) -> str | None:
    """
    Pull the host-side mapped port for *container_port* out of an instance dict.

    VAST.AI stores port mappings under the 'ports' key as a Docker-style dict:
        {"7860/tcp": [{"HostIp": "", "HostPort": "51234"}], ...}
    Returns the HostPort string, or None if not mapped / not yet assigned.
    """
    ports: dict = instance.get("ports") or {}
    tcp_key = f"{container_port}/tcp"
    mappings = ports.get(tcp_key) or []
    if mappings:
        return mappings[0].get("HostPort")
    return None


def main() -> None:
    api_key = get_api_key()
    if not api_key:
        print(
            "ERROR: No VAST API key found.\n"
            "  Option 1:  export VAST_API_KEY=your_key\n"
            "  Option 2:  pip install vastai && vastai set api-key your_key\n"
            "Get your key from: https://console.vast.ai/manage-keys/"
        )
        sys.exit(1)

    vast = VastAI(api_key=api_key)

    print("Fetching your VAST.AI instances…\n")
    instances: list[dict] = vast.show_instances()

    if not instances:
        print("No instances found.  Rent one at https://cloud.vast.ai")
        return

    found_any = False

    for inst in instances:
        inst_id     = inst.get("id", "?")
        status      = inst.get("status", "unknown")
        label       = inst.get("label") or ""
        gpu_name    = inst.get("gpu_name", "?")
        num_gpus    = inst.get("num_gpus", 1)
        public_ip   = inst.get("public_ipaddr", "").strip()
        dph         = inst.get("dph_total", 0.0)

        host_port   = extract_host_port(inst, GRADIO_PORT)

        label_str   = f"  label    : {label}" if label else ""
        gpu_str     = f"{num_gpus}× {gpu_name}"
        cost_str    = f"${dph:.3f}/hr"

        print(f"Instance {inst_id}  [{status}]  {gpu_str}  {cost_str}")
        if label_str:
            print(label_str)

        if status != "running":
            print("  → Not running yet; check back shortly.\n")
            continue

        if not public_ip:
            print("  → No public IP assigned yet.\n")
            continue

        if host_port:
            url = f"http://{public_ip}:{host_port}"
            print(f"  Gradio UI → {url}")
            found_any = True
        else:
            # Port 7860 not mapped — show what IS mapped so the user can check
            ports = inst.get("ports") or {}
            if ports:
                print(f"  Port {GRADIO_PORT} not mapped.  Mapped ports: {list(ports.keys())}")
            else:
                print(
                    f"  Port {GRADIO_PORT} not mapped.  "
                    "Make sure you added 7860 under 'Exposed ports' when renting."
                )
        print()

    if not found_any:
        print(
            "No reachable Gradio UI found.\n"
            "Check that your instance is running and port 7860 was exposed."
        )


if __name__ == "__main__":
    main()
