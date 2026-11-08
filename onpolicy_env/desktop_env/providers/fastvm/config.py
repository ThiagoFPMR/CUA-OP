"""FastVM provider settings, read from environment variables (defaults below).

FASTVM_API_KEY is required. FASTVM_OSWORLD_SNAPSHOT_ID names the snapshot VMs start
from, which must run the OSWorld in-VM server on port 5000.
"""
from __future__ import annotations

import os

DEFAULT_MACHINE_TYPE = os.environ.get("FASTVM_MACHINE_TYPE", "c4m8")
OSWORLD_SNAPSHOT_ID = os.environ.get("FASTVM_OSWORLD_SNAPSHOT_ID") or None
DISK_GIB = int(os.environ.get("FASTVM_DISK_GIB", "0")) or None  # 0/unset → omit
LAUNCH_TIMEOUT_SEC = float(os.environ.get("FASTVM_LAUNCH_TIMEOUT_SEC", "300"))
READY_TIMEOUT_SEC = float(os.environ.get("FASTVM_READY_TIMEOUT_SEC", "300"))
SKIP_READY_CHECK = os.environ.get("FASTVM_SKIP_READY_CHECK") == "1"

# OSWorld in-VM service ports. Kept constant across providers so the
# DesktopEnv controllers don't care which backend we're on.
SERVER_PORT = 5000
CHROMIUM_PORT = 9222
VNC_PORT = 8006
VLC_PORT = 8080
NOVNC_PORT = 5910
