import ipaddress
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time

from onpolicy_env.desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.vmware.VMwareProvider")
logger.setLevel(logging.INFO)

WAIT_TIME = 3
# Upper bound on any single vmrun call; `getGuestIPAddress -wait` is the slow one.
VMRUN_TIMEOUT = 300
# How long to keep retrying a power-on, and to wait for the guest to report an IP.
# A healthy guest does both in well under a minute.
START_TIMEOUT = 300
IP_TIMEOUT = 600
# Free space kept beyond the VM's memory file before powering it on.
FREE_SPACE_MARGIN = 2 * 2**30


def get_vmrun_type(return_list=False):
    if platform.system() == 'Windows' or platform.system() == 'Linux':
        if return_list:
            return ['-T', 'ws']
        else:
            return '-T ws'
    elif platform.system() == 'Darwin':  # Darwin is the system name for macOS
        if return_list:
            return ['-T', 'fusion']
        else:
            return '-T fusion'
    else:
        raise Exception("Unsupported operating system")


class VMwareError(RuntimeError):
    """A `vmrun` call failed, timed out, or left the VM unusable."""


class VMwareProvider(Provider):
    @staticmethod
    def _execute_command(command: list, return_output=False, timeout=VMRUN_TIMEOUT):
        """Run `vmrun` to completion and fail loudly if it did not succeed.

        vmrun reports failures on stdout as ``Error: ...`` (stderr only carries
        its AppLoader banner), sometimes with a zero exit code, so both are checked.

        Output goes to temp files, not pipes: `vmrun start` spawns vmware-vmx,
        which inherits vmrun's stdout/stderr and outlives it. Reading a pipe
        would wait for vmware-vmx to close it, so the call would hang until the
        timeout although vmrun itself exited once the VM was up.
        """
        with tempfile.TemporaryFile("w+", encoding="utf-8") as out, \
                tempfile.TemporaryFile("w+", encoding="utf-8") as err:
            try:
                result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                        timeout=timeout)
            except subprocess.TimeoutExpired as e:
                raise VMwareError(f"{' '.join(command)} timed out after {timeout}s") from e
            out.seek(0)
            err.seek(0)
            output, error = out.read().strip(), err.read().strip()

        if result.returncode != 0 or output.startswith("Error"):
            raise VMwareError(
                f"{' '.join(command)} failed (rc={result.returncode}): {output or error}")
        return output if return_output else None

    @staticmethod
    def _is_running(path_to_vm: str) -> bool:
        output = VMwareProvider._execute_command(
            ["vmrun"] + get_vmrun_type(return_list=True) + ["list"], return_output=True)
        normalized_path_to_vm = os.path.abspath(os.path.normpath(path_to_vm))
        return any(os.path.abspath(os.path.normpath(line)) == normalized_path_to_vm
                   for line in output.splitlines())

    @staticmethod
    def _check_free_space(path_to_vm: str):
        """Refuse to power on when the VM's memory file would not fit on disk.

        VMware backs guest RAM with a file next to the .vmx. Short of room, it
        does not fail the power-on: it pauses the VM behind a dialog, which
        headless automation never answers, and the guest hangs until space frees.
        """
        memsize_mb = 4096
        with open(path_to_vm, "r", encoding="utf-8", errors="replace") as f:
            match = re.search(r'^memsize\s*=\s*"(\d+)"', f.read(), re.MULTILINE)
        if match:
            memsize_mb = int(match.group(1))
        needed = memsize_mb * 2**20 + FREE_SPACE_MARGIN
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(path_to_vm))).free
        if free < needed:
            raise VMwareError(
                f"only {free / 2**30:.1f} GiB free next to {path_to_vm}; powering it on needs "
                f"{needed / 2**30:.1f} GiB ({memsize_mb} MiB memory file + headroom)")

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str):
        print("Starting VMware VM...")
        logger.info("Starting VMware VM...")

        deadline = time.monotonic() + START_TIMEOUT
        while True:
            if self._is_running(path_to_vm):
                logger.info("VM is running.")
                return
            if time.monotonic() > deadline:
                raise VMwareError(f"{path_to_vm} did not start within {START_TIMEOUT}s")

            self._check_free_space(path_to_vm)
            logger.info("Starting VM...")
            _command = ["vmrun"] + get_vmrun_type(return_list=True) + ["start", path_to_vm]
            if headless:
                _command.append("nogui")
            try:
                VMwareProvider._execute_command(_command)
            except VMwareError as e:
                logger.error(str(e))
            time.sleep(WAIT_TIME)

    def get_ip_address(self, path_to_vm: str) -> str:
        logger.info("Getting VMware VM IP address...")
        deadline = time.monotonic() + IP_TIMEOUT
        while True:
            try:
                output = VMwareProvider._execute_command(
                    ["vmrun"] + get_vmrun_type(return_list=True) + ["getGuestIPAddress", path_to_vm, "-wait"],
                    return_output=True
                )
                ip = output.splitlines()[-1].strip() if output else ""
                ipaddress.ip_address(ip)  # anything else is vmrun talking, not an address
                logger.info(f"VMware VM IP address: {ip}")
                return ip
            except (VMwareError, ValueError) as e:
                if time.monotonic() > deadline:
                    raise VMwareError(f"no guest IP for {path_to_vm} within {IP_TIMEOUT}s: {e}") from e
                logger.error(e)
                time.sleep(WAIT_TIME)
                logger.info("Retrying to get VMware VM IP address...")

    def save_state(self, path_to_vm: str, snapshot_name: str):
        logger.info("Saving VMware VM state...")
        VMwareProvider._execute_command(
            ["vmrun"] + get_vmrun_type(return_list=True) + ["snapshot", path_to_vm, snapshot_name])
        time.sleep(WAIT_TIME)  # Wait for the VM to save

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        logger.info(f"Reverting VMware VM to snapshot: {snapshot_name}...")
        VMwareProvider._execute_command(
            ["vmrun"] + get_vmrun_type(return_list=True) + ["revertToSnapshot", path_to_vm, snapshot_name])
        time.sleep(WAIT_TIME)  # Wait for the VM to revert
        return path_to_vm

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        # Note: region parameter is ignored for VMware provider
        # but kept for interface consistency with other providers
        logger.info("Stopping VMware VM...")
        if not self._is_running(path_to_vm):
            logger.info("VM is not running; nothing to stop.")
            return
        _command = ["vmrun"] + get_vmrun_type(return_list=True) + ["stop", path_to_vm]
        try:
            VMwareProvider._execute_command(_command, timeout=120)
        except VMwareError as e:
            # A guest that is hung, or paused behind a dialog, ignores a soft stop.
            logger.warning(f"{e}; forcing a hard stop.")
            VMwareProvider._execute_command(_command + ["hard"], timeout=120)
        time.sleep(WAIT_TIME)  # Wait for the VM to stop
