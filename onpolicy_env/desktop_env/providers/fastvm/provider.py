"""FastVM implementation of OSWorld's `Provider`, over the `fastvm` SDK.

`path_to_vm` is the FastVM VM id, and reverting to a snapshot replaces the VM with
a new one. Addresses use the Docker provider's `<host>:<server>:<chromium>:<vnc>:<vlc>`.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from onpolicy_env.desktop_env.providers.base import Provider
from onpolicy_env.desktop_env.providers.fastvm.config import (
    CHROMIUM_PORT,
    LAUNCH_TIMEOUT_SEC,
    READY_TIMEOUT_SEC,
    SERVER_PORT,
    SKIP_READY_CHECK,
    VLC_PORT,
    VNC_PORT,
)
from onpolicy_env.desktop_env.providers.fastvm.client import get_client

logger = logging.getLogger("desktopenv.providers.fastvm.FastvmProvider")
logger.setLevel(logging.INFO)

_RUNNING = "running"
_TERMINAL_FAILURE = {"error", "stopped", "deleting"}


def _osworld_firewall() -> dict:
    """Open the ports OSWorld agents & noVNC need, IPv6-any source.

    FastVM only accepts IPv6 CIDRs; omitting `source_cidrs` means any-source.
    We keep `mode=open` for the trial demo — switch to `restricted` once
    we pin down the caller's v6 prefix.
    """
    return {"mode": "open"}


def _bracket(host: str) -> str:
    """Wrap literal IPv6 addresses in `[...]` so URLs parse correctly.

    Hostnames and IPv4 pass through. Double-bracketing is a no-op.
    """
    if not host:
        return host
    if host.startswith("[") and host.endswith("]"):
        return host
    if ":" in host:
        return f"[{host}]"
    return host


def _format_vm_ip_ports(host: str) -> str:
    """Emit the `host:server:chromium:vnc:vlc` tuple DesktopEnv expects."""
    return f"{_bracket(host)}:{SERVER_PORT}:{CHROMIUM_PORT}:{VNC_PORT}:{VLC_PORT}"


def _wait_for_server(host: str, timeout: float) -> None:
    """Poll the in-VM Flask server until `/screenshot` returns 200.

    Raises TimeoutError on budget exhaustion so the caller can clean up.
    """
    url = f"http://{_bracket(host)}:{SERVER_PORT}/screenshot"
    deadline = time.monotonic() + timeout
    last_err: Optional[str] = None
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=(5, 5))
            if r.status_code == 200:
                logger.info(f"OSWorld in-VM server is ready at {url}")
                return
            last_err = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001 — accept any connect-time error
            last_err = type(e).__name__
        logger.info(f"Waiting for OSWorld server at {url} (last: {last_err})…")
        time.sleep(3)
    raise TimeoutError(
        f"OSWorld in-VM server at {url} not ready after {timeout:.0f}s "
        f"(last={last_err}). If the base image lacks `osworld_server`, "
        f"set FASTVM_SKIP_READY_CHECK=1 to smoke-test the control plane only."
    )


class FastvmProvider(Provider):
    def __init__(self, region: Optional[str] = None) -> None:
        super().__init__(region=region)
        self._client = get_client()

    # ------------------------------------------------------------------ #
    #                            Lifecycle                               #
    # ------------------------------------------------------------------ #

    def start_emulator(self, path_to_vm: str, headless: bool, *args, **kwargs):
        """Ensure the VM is running.

        The manager's `get_vm_path` and our `revert_to_snapshot` already
        launch-and-wait, so this is normally a cheap retrieve. We keep it
        around to honour the Provider contract and to handle pathological
        cases where a VM returned from cache is mid-provisioning.
        """
        logger.info(f"Ensuring FastVM {path_to_vm} is running…")
        vm = self._client.vms.retrieve(path_to_vm)
        if vm.status == _RUNNING:
            return
        if vm.status in _TERMINAL_FAILURE:
            raise RuntimeError(f"FastVM {path_to_vm} is in terminal state {vm.status!r}")
        # Transitional (provisioning / unknown) — poll until running.
        self._client.wait_for_vm_ready(path_to_vm, timeout=LAUNCH_TIMEOUT_SEC)

    def get_ip_address(self, path_to_vm: str) -> str:
        """Return `<ipv6>:<server>:<chromium>:<vnc>:<vlc>` and wait for :5000."""
        logger.info(f"Resolving FastVM {path_to_vm} address…")
        vm = self._client.vms.retrieve(path_to_vm)
        host = vm.public_ipv6 or ""
        if not host:
            raise RuntimeError(
                f"FastVM {path_to_vm} has no publicIpv6 yet (status={vm.status!r}). "
                f"If the VM just booted, wait a few seconds and retry."
            )
        # Banner mirrors the AWS provider so ops get a copy-pasteable URL.
        novnc_url = f"http://{_bracket(host)}:5910/vnc.html"
        logger.info("=" * 80)
        logger.info(f"🖥️  VNC Web Access URL: {novnc_url}")
        logger.info(f"📡 Public IPv6: {host}")
        logger.info(f"🆔 VM ID: {path_to_vm}")
        logger.info("=" * 80)
        print(f"\n🌐 VNC Web Access URL: {novnc_url}")
        print("📍 Open the above in a browser for remote desktop access\n")

        if not SKIP_READY_CHECK:
            _wait_for_server(host, READY_TIMEOUT_SEC)
        else:
            logger.warning("FASTVM_SKIP_READY_CHECK=1 — not probing :5000")

        return _format_vm_ip_ports(host)

    # ------------------------------------------------------------------ #
    #                            Snapshots                               #
    # ------------------------------------------------------------------ #

    def save_state(self, path_to_vm: str, snapshot_name: str):
        logger.info(f"Snapshotting FastVM {path_to_vm} as {snapshot_name!r}…")
        snap = self._client.snapshots.create(vm_id=path_to_vm, name=snapshot_name)
        # Poll until ready — FastVM snapshots are async. Reuse a short polling
        # loop rather than pulling in another dependency.
        deadline = time.monotonic() + LAUNCH_TIMEOUT_SEC
        while True:
            current = next(
                (s for s in self._client.snapshots.list() if s.id == snap.id),
                None,
            )
            if current is None:
                raise RuntimeError(f"Snapshot {snap.id} disappeared while polling")
            if current.status == "ready":
                logger.info(f"Snapshot {snap.id} ready.")
                return snap.id
            if current.status == "error":
                raise RuntimeError(f"Snapshot {snap.id} entered error state")
            if time.monotonic() > deadline:
                raise TimeoutError(f"Snapshot {snap.id} not ready after {LAUNCH_TIMEOUT_SEC}s")
            time.sleep(2)

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        """Delete the existing VM and launch a fresh one from `snapshot_name`.

        `snapshot_name` here is a FastVM snapshot *id* — naming follows the
        AWS provider's `create_image` → `ami-…` convention where the AMI id
        gets threaded through this same parameter.
        """
        logger.info(f"Reverting: tearing down {path_to_vm}, restoring from {snapshot_name}…")
        try:
            self._client.vms.delete(path_to_vm)
        except Exception as e:  # noqa: BLE001
            # If the VM is already gone (e.g. TTL reaped it), don't block the revert.
            logger.warning(f"Could not delete old VM {path_to_vm}: {e}")

        vm = self._client.vms.launch(
            snapshot_id=snapshot_name,
            firewall=_osworld_firewall(),
            name=f"osworld-{int(time.time())}",
            wait_timeout=LAUNCH_TIMEOUT_SEC,
        )
        logger.info(f"Restored FastVM {vm.id} from snapshot {snapshot_name}")
        return vm.id

    # ------------------------------------------------------------------ #
    #                            Teardown                                #
    # ------------------------------------------------------------------ #

    def stop_emulator(self, path_to_vm: str, *args, **kwargs):
        logger.info(f"Deleting FastVM {path_to_vm}…")
        try:
            self._client.vms.delete(path_to_vm)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to delete FastVM {path_to_vm}: {e}")
            raise
