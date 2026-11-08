"""Check a task set before a run (`onpolicy-preflight`).

Validates every task's setup steps and downloads the files they need into the host
cache; with --vm, also runs each task's setup in a VM and screenshots the result.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import shutil
import sys
import time
import uuid
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import Manager, Process

import requests

from onpolicy_env.desktop_env.controllers.setup import (DOWNLOAD_HEADERS, OFFLINE_ENV_VAR,
                                           SetupController, download_cache_path,
                                           fetch_to_cache)

logger = logging.getLogger("desktopenv.preflight")
LOG_FORMAT = "[%(asctime)s %(levelname)s %(processName)s] %(message)s"

# `DesktopEnv` caches downloads under this directory, relative to the working
# directory, unless it is built with another `cache_dir` (the runner does not).
DEFAULT_CACHE_DIR = "cache"


@dataclass
class Task:
    domain: str
    example_id: str
    config_path: str
    example: dict | None = None  # None when the file would not load

    @property
    def key(self) -> str:
        return f"{self.domain}/{self.example_id}"


@dataclass
class Report:
    """Problems found, per task key. Errors fail a task; warnings do not."""
    errors: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    warnings: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def error(self, key: str, msg: str) -> None:
        self.errors[key].append(msg)

    def warn(self, key: str, msg: str) -> None:
        self.warnings[key].append(msg)

    def log(self, total: int) -> None:
        for key in sorted(set(self.errors) | set(self.warnings)):
            for msg in self.errors.get(key, []):
                logger.error("%s: %s", key, msg)
            for msg in self.warnings.get(key, []):
                logger.warning("%s: %s", key, msg)
        logger.info("Preflight: %d/%d tasks OK, %d failed, %d with warnings.",
                    total - len(self.errors), total, len(self.errors),
                    len(set(self.warnings) - set(self.errors)))


# --------------------------------------------------------------------------- tasks

def select_tasks(meta_path: str, domains: str = "") -> list[tuple[str, str]]:
    """(domain, example_id) pairs from a test_all-style manifest.

    ``domains`` is a comma-separated subset; empty means all of them.
    """
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    requested = [d.strip() for d in domains.split(",") if d.strip()]
    if requested:
        missing = [d for d in requested if d not in meta]
        if missing:
            raise KeyError(f"domain(s) {missing} not in {meta_path} "
                           f"(available: {sorted(meta)})")
        meta = {d: meta[d] for d in requested}
    return [(domain, example_id)
            for domain, examples in meta.items() for example_id in examples]


def load_tasks(base_dir: str, pairs: list[tuple[str, str]], report: Report) -> list[Task]:
    tasks = []
    for domain, example_id in pairs:
        task = Task(domain, example_id,
                    os.path.join(base_dir, "examples", domain, f"{example_id}.json"))
        try:
            with open(task.config_path, "r", encoding="utf-8") as f:
                task.example = json.load(f)
            if not isinstance(task.example, dict):
                raise ValueError("top level is not an object")
        except (OSError, ValueError) as exc:
            task.example = None
            report.error(task.key, f"cannot load {task.config_path}: {exc}")
        tasks.append(task)
    return tasks


# ------------------------------------------------------------------ static checks

def check_task(task: Task, report: Report) -> None:
    """Everything that can be wrong with a task without the network or a VM."""
    ex = task.example
    key = task.key
    if not isinstance(ex.get("instruction"), str) or not ex["instruction"].strip():
        report.error(key, "no instruction")
    if not ex.get("id"):
        report.error(key, "no id (the env keys its download cache on it)")
    elif ex["id"] != task.example_id:
        report.warn(key, f"id {ex['id']!r} differs from the file name")

    config = ex.get("config", [])
    if not isinstance(config, list):
        report.error(key, "config is not a list of setup steps")
        config = []
    downloaded: set[str] = set()
    for i, step in enumerate(config, 1):
        where = f"setup step {i}"
        if not isinstance(step, dict) or "type" not in step or "parameters" not in step:
            report.error(key, f"{where}: needs a 'type' and 'parameters'")
            continue
        step_type, params = step["type"], step["parameters"]
        fn = getattr(SetupController, f"_{step_type}_setup", None)
        if fn is None:
            report.error(key, f"{where}: unknown step type {step_type!r}")
            continue
        if not isinstance(params, dict):
            report.error(key, f"{where} ({step_type}): parameters is not an object")
            continue
        try:
            inspect.signature(fn).bind(None, **params)
        except TypeError as exc:
            report.error(key, f"{where} ({step_type}): bad parameters: {exc}")
            continue
        if step_type == "download":
            for f in params["files"]:
                url, path = f.get("url", ""), f.get("path", "")
                if not str(url).startswith(("http://", "https://")):
                    report.error(key, f"{where}: download url {url!r} is not http(s)")
                if not str(path).startswith("/"):
                    report.error(key, f"{where}: download path {path!r} is not absolute")
                downloaded.add(path)
        # A file an app is started on must come from an earlier download or
        # already be in the VM image; only a rehearsal can tell the latter.
        if step_type == "open":
            files = [params["path"]]
        elif step_type == "launch":
            command = params["command"]
            args = command.split() if isinstance(command, str) else command
            files = [a for a in args[1:] if isinstance(a, str) and a.startswith("/")]
        else:
            files = []
        for path in files:
            if path not in downloaded:
                report.warn(key, f"{where} ({step_type}) uses {path}, which no earlier "
                                 "download provides; it must already be in the VM image")


# ------------------------------------------------------------------------- assets

_MAGIC = {
    ".jpg": (b"\xff\xd8\xff",), ".jpeg": (b"\xff\xd8\xff",),
    ".png": (b"\x89PNG\r\n\x1a\n",), ".gif": (b"GIF87a", b"GIF89a"),
    ".pdf": (b"%PDF",), ".ogg": (b"OggS",), ".mkv": (b"\x1a\x45\xdf\xa3",),
    ".webm": (b"\x1a\x45\xdf\xa3",),
    # Legacy Office files are OLE compound documents.
    ".doc": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",), ".xls": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),
    ".ppt": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),
}
_ZIP = {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp", ".zip"}
# The part every OOXML file of that kind carries.
_OOXML_PART = {".docx": "word/", ".xlsx": "xl/", ".pptx": "ppt/"}
_ISO_MEDIA = {".mp4", ".m4v", ".mov", ".m4a"}
_IMAGES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
_MARKUP = {".html", ".htm", ".xml", ".svg"}


def _sniff(path: str, head: bytes) -> str:
    """What a file's content says it is, for error messages."""
    for ext, magics in _MAGIC.items():
        if head.startswith(magics):
            return "a legacy Office (OLE) file" if ext in (".doc", ".xls", ".ppt") \
                else f"a {ext.lstrip('.').upper()} file"
    if head[4:8] == b"ftyp":
        return "an MP4/QuickTime file"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
        except zipfile.BadZipFile:
            return "a damaged zip file"
        for ext, part in _OOXML_PART.items():
            if any(n.startswith(part) for n in names):
                return f"a {ext} file"
        return "an OpenDocument file" if "mimetype" in names else "a zip file"
    return "something else"


def validate_asset(path: str, name: str) -> str | None:
    """Why the file at ``path`` is not a usable ``name``, or None if it is.

    Catches what a download that "succeeded" can still hand back: an empty body,
    an HTML error or consent page, and a truncated file.
    """
    if os.path.getsize(path) == 0:
        return "empty file"
    ext = os.path.splitext(name)[1].lower()
    with open(path, "rb") as f:
        head = f.read(512)
    lead = head.lstrip()[:14].lower()
    if ext not in _MARKUP and lead.startswith((b"<!doctype html", b"<html")):
        return f"is an HTML page, not a {ext or 'data'} file"
    if (ext in _MAGIC and not head.startswith(_MAGIC[ext])) \
            or (ext in _ISO_MEDIA and head[4:8] != b"ftyp") \
            or (ext in _ZIP and not head.startswith(b"PK\x03\x04")):
        return f"is {_sniff(path, head)}, but the task names it {ext}"
    if ext in _ZIP:
        try:
            with zipfile.ZipFile(path) as z:
                bad = z.testzip()
                names = z.namelist()
        except zipfile.BadZipFile as exc:
            return f"not a valid {ext} (zip) file: {exc}"
        if bad is not None:
            return f"corrupt {ext}: member {bad!r} fails its CRC"
        part = _OOXML_PART.get(ext)
        if part and not any(n.startswith(part) for n in names):
            return f"is {_sniff(path, head)}, but the task names it {ext}"
    if ext in _IMAGES:
        from PIL import Image
        try:
            with Image.open(path) as im:
                im.load()  # decodes every pixel, so truncation shows
        except Exception as exc:  # noqa: BLE001
            return f"image does not decode: {exc}"
    return None


@dataclass
class _Target:
    task_key: str
    cache_path: str
    name: str  # basename of the VM path; its extension says what the file is


def _prefetch_url(url: str, targets: list[_Target], check_only: bool) -> dict[str, str]:
    """Make every target a valid copy of ``url``; return errors by task key.

    The URL is fetched at most once and copied to the other tasks' cache paths.
    """
    errors: dict[str, str] = {}
    source = None
    for t in targets:
        if not os.path.exists(t.cache_path):
            continue
        problem = validate_asset(t.cache_path, t.name)
        if problem is None:
            source = source or t.cache_path
        else:
            logger.warning("Discarding bad cached %s: %s", t.cache_path, problem)
            os.remove(t.cache_path)

    if source is None:
        if check_only:
            return {t.task_key: f"{url} is not cached" for t in targets}
        first = targets[0]
        try:
            fetch_to_cache(url, first.cache_path)
        except requests.RequestException as exc:
            return {t.task_key: f"download failed: {exc}" for t in targets}
        source = first.cache_path

    for t in targets:
        if not os.path.exists(t.cache_path):
            if check_only:
                errors[t.task_key] = f"{url} is not cached at {t.cache_path}"
                continue
            os.makedirs(os.path.dirname(t.cache_path), exist_ok=True)
            tmp = f"{t.cache_path}.part.{uuid.uuid4().hex}"
            shutil.copyfile(source, tmp)
            os.replace(tmp, t.cache_path)
        problem = validate_asset(t.cache_path, t.name)
        if problem is not None:
            os.remove(t.cache_path)
            errors[t.task_key] = f"{url} {problem}"
    return errors


def prefetch_assets(tasks: list[Task], cache_dir: str, report: Report,
                    workers: int = 4, check_only: bool = False) -> None:
    """Fill (or with ``check_only``, just audit) the download cache for ``tasks``."""
    by_url: dict[str, list[_Target]] = defaultdict(list)
    for task in tasks:
        for step in task.example.get("config", []):
            if step["type"] != "download":
                continue
            for f in step["parameters"]["files"]:
                by_url[f["url"]].append(_Target(
                    task.key,
                    download_cache_path(os.path.join(cache_dir, task.example["id"]),
                                        f["url"], f["path"]),
                    os.path.basename(f["path"])))
    if not by_url:
        return
    logger.info("%s %d download assets for %d tasks under %s/.",
                "Checking" if check_only else "Fetching", len(by_url),
                len({t.task_key for ts in by_url.values() for t in ts}), cache_dir)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_prefetch_url, url, targets, check_only): url
                   for url, targets in by_url.items()}
        for future, url in futures.items():
            try:
                for key, msg in future.result().items():
                    report.error(key, msg)
            except Exception as exc:  # noqa: BLE001
                for t in by_url[url]:
                    report.error(t.task_key, f"{url}: {type(exc).__name__}: {exc}")


def check_live_urls(tasks: list[Task], report: Report, workers: int = 8) -> None:
    """Warn about pages `chrome_open_tabs` will fail to open.

    Fetched from the host, which is only a proxy for what the VM will see, and
    some sites refuse clients that are not browsers; hence warnings, not errors.
    """
    by_url: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        for step in task.example.get("config", []):
            if step["type"] == "chrome_open_tabs":
                for url in step["parameters"]["urls_to_open"]:
                    by_url[url].append(task.key)
    if not by_url:
        return

    def probe(url: str) -> str | None:
        try:
            resp = requests.get(url, headers=DOWNLOAD_HEADERS, timeout=20, stream=True)
            resp.close()
            return None if resp.status_code < 400 else f"HTTP {resp.status_code}"
        except requests.RequestException as exc:
            return f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for url, problem in zip(by_url, pool.map(probe, by_url)):
            if problem is not None:
                for key in by_url[url]:
                    report.warn(key, f"page the task opens in Chrome, {url}, "
                                     f"did not load from the host ({problem})")


def run_preflight(base_dir: str, pairs: list[tuple[str, str]],
                  cache_dir: str = DEFAULT_CACHE_DIR, workers: int = 4,
                  check_only: bool = False, live_urls: bool = True) -> tuple[list[Task], Report]:
    """Static checks, then assets, for the tasks named by ``pairs``."""
    report = Report()
    tasks = load_tasks(base_dir, pairs, report)
    loaded = [t for t in tasks if t.example is not None]
    for task in loaded:
        check_task(task, report)
    # Only tasks whose steps are well-formed can have their downloads walked.
    sound = [t for t in loaded if t.key not in report.errors]
    prefetch_assets(sound, cache_dir, report, workers=workers, check_only=check_only)
    if live_urls:
        check_live_urls(sound, report)
    report.log(len(tasks))
    return tasks, report


# ---------------------------------------------------------------------- rehearsal

class _SetupLog(logging.Handler):
    """Collects what the setup steps log while one task resets.

    `launch`, `execute` and `activate_window` log a failure and carry on, so a
    setup that "succeeds" can still leave the task on the wrong screen; the
    rehearsal counts their errors as failures.
    """

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self, level: int) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno == level]


def _rehearsal_worker(task_queue, results, args) -> None:
    """Run each queued task's setup on one env; record (error, warnings) for it."""
    from onpolicy_env.env import build_env

    setup_logger = logging.getLogger("desktopenv.setup")
    env = None
    try:
        env = build_env(args)
        while True:
            try:
                key, config_path = task_queue.get(timeout=5)
            except Exception:  # noqa: BLE001 - queue drained
                break
            with open(config_path, "r", encoding="utf-8") as f:
                example = json.load(f)
            started = time.time()
            captured = _SetupLog()
            setup_logger.addHandler(captured)
            try:
                env.reset(task_config=example)
                time.sleep(args.settle)
                shot = env.controller.get_screenshot()
                if shot:
                    out = os.path.join(args.rehearsal_dir, f"{key}.png")
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with open(out, "wb") as f:
                        f.write(shot)
                errors = captured.messages(logging.ERROR)
                error = "; ".join(errors) if errors else None
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
            finally:
                setup_logger.removeHandler(captured)
            results[key] = (error, captured.messages(logging.WARNING))
            if error is None:
                logger.info("Rehearsal OK: %s (%.0fs)", key, time.time() - started)
            else:
                # Its steps may have changed the VM; revert before the next task.
                env.is_environment_used = True
                logger.error("Rehearsal FAILED: %s: %s", key, error)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not bring up an env: %s: %s", type(exc).__name__, exc)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001
                logger.error("env close failed: %s", exc)


def rehearse(tasks: list[Task], args, report: Report) -> None:
    """Run every task's setup in real VMs, ``args.num_envs`` at a time."""
    logger.info("Rehearsing setup of %d tasks on %d env(s); screenshots go to %s/.",
                len(tasks), args.num_envs, args.rehearsal_dir)
    with Manager() as manager:
        task_queue = manager.Queue()
        results = manager.dict()
        for task in tasks:
            task_queue.put((task.key, task.config_path))
        procs = [Process(target=_rehearsal_worker, args=(task_queue, results, args),
                         name=f"RehearsalEnv-{i + 1}") for i in range(args.num_envs)]
        for proc in procs:
            proc.start()
        try:
            for proc in procs:
                proc.join()
        except KeyboardInterrupt:
            # The workers got the same Ctrl-C and are shutting their VMs down.
            logger.info("Interrupted; waiting for the VMs to stop.")
            for proc in procs:
                proc.join(timeout=90)
            raise
        done = dict(results)
    for task in tasks:
        if task.key not in done:
            report.error(task.key, "setup not rehearsed (no env was left to run it)")
            continue
        error, warnings = done[task.key]
        if error is not None:
            report.error(task.key, f"setup failed in the VM: {error}")
        for msg in warnings:
            report.warn(task.key, f"setup in the VM: {msg}")


# --------------------------------------------------------------------------- main

def config() -> argparse.Namespace:
    from onpolicy_env.env import add_env_args

    p = argparse.ArgumentParser(description="Validate a task set and pre-download its assets")
    p.add_argument("--test_config_base_dir", type=str, default="evaluation_examples")
    p.add_argument("--test_all_meta_path", type=str,
                   default="evaluation_examples/test_all.json")
    p.add_argument("--domains", type=str, default="",
                   help="comma-separated subset of domains; empty means all")
    p.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR,
                   help="host download cache; must match the one the runs use")
    p.add_argument("--download_workers", type=int, default=4)
    p.add_argument("--check_only", action="store_true",
                   help="audit the cache without downloading anything")
    p.add_argument("--skip_live_urls", action="store_true",
                   help="do not probe the pages chrome_open_tabs opens")

    p.add_argument("--vm", action="store_true",
                   help="also run every task's setup in real VMs (slow)")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--settle", type=float, default=10.0,
                   help="seconds to wait after setup before the screenshot")
    p.add_argument("--rehearsal_dir", type=str, default="results_preflight",
                   help="where the rehearsal writes each task's starting screenshot")
    add_env_args(p)
    p.add_argument("--log_level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return p.parse_args()


def main() -> int:
    args = config()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    root.setLevel(getattr(logging, args.log_level.upper()))
    root.addHandler(handler)

    pairs = select_tasks(args.test_all_meta_path, args.domains)
    tasks, report = run_preflight(args.test_config_base_dir, pairs,
                                  cache_dir=args.cache_dir,
                                  workers=args.download_workers,
                                  check_only=args.check_only,
                                  live_urls=not args.skip_live_urls)
    if args.vm:
        if args.cache_dir != DEFAULT_CACHE_DIR:
            logger.error("--vm needs the default --cache_dir: the env reads %s/.",
                         DEFAULT_CACHE_DIR)
            return 1
        ready = [t for t in tasks if t.key not in report.errors]
        # The rehearsal must use the cache the checks just filled, like a run.
        os.environ[OFFLINE_ENV_VAR] = "1"
        rehearse(ready, args, report)
        report.log(len(tasks))
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
