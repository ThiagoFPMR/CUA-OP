"""Run desktop tasks on-policy against the profiling server (`onpolicy-run`).

One worker per VM. Each task is one episode on the server: the agent's requests go
to the episode's URL, and each agent decision is announced as a step.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shutil
import signal
import sys
import time
import uuid
from contextlib import contextmanager
from multiprocessing import Manager, Process, current_process

from onpolicy_env.env import add_env_args, build_env
from onpolicy_env.preflight import run_preflight, select_tasks
from onpolicy_env.desktop_env.controllers.setup import OFFLINE_ENV_VAR
from onpolicy_env.mm_agents.holo3_agent import Holo3Agent
from onpolicy_env.mm_agents.qwen import QwenAgent
from moe_profiler.server.client import ProfilingClient

logger = logging.getLogger("desktopenv.experiment")
LOG_FORMAT = "[%(asctime)s %(levelname)s %(processName)s] %(message)s"

processes: list[Process] = []
# Workers are forked, so they inherit both `processes` and the parent's signal
# handlers. The pid taken here is the parent's in the parent and, in a child,
# still the parent's — which is how a handler tells which side it is running on.
_MAIN_PID = os.getpid()
_shutting_down = False

# How long the parent lets a worker finish shutting its VM down before killing it.
SHUTDOWN_GRACE = 90.0
# A worker gives up only if `/episode/start` keeps failing: one refusal is a task
# lost, a run of them means the server is gone and there is nothing to profile.
MAX_CONSECUTIVE_START_FAILURES = 5
# Likewise for episodes: an env that fails this many in a row is broken (VM that
# will not boot, guest it cannot reach), and every further task it pulls would
# fail in minutes and be lost. The worker hands its task back and exits instead.
MAX_CONSECUTIVE_EPISODE_FAILURES = 3
# Free disk each VMware env needs to boot: a 4 GiB memory file plus disk deltas.
# VMware pauses a VM it lacks room for rather than failing, so check up front.
VMWARE_FREE_BYTES_PER_ENV = 6 * 2**30


@contextmanager
def _episode_log(result_dir: str):
    """Also write everything logged during an episode, env and agent included,
    to ``runner.log`` next to its screenshots, so a failed episode says why."""
    handler = logging.FileHandler(os.path.join(result_dir, "runner.log"))
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()


def run_episode(client: ProfilingClient, episode_id: str, env, agent,
                example: dict, domain: str, args, result_dir: str) -> str:
    """One on-policy rollout, framed for the server.

    Reset the environment and agent, then act until the task is done or the step
    budget is spent, pushing the step before each decision. Screenshots and a
    trajectory log are saved for debugging; the routing statistics themselves are
    written server-side.

    Returns:
        The stop reason: ``complete`` or ``max_steps``.
    """
    instruction = example["instruction"]
    env.reset(task_config=example)
    try:
        agent.reset(logger, vm_ip=env.vm_ip)
    except Exception:
        agent.reset(vm_ip=env.vm_ip)

    time.sleep(60)  # Let the environment come up.
    obs = env._get_obs()
    done = False
    step_idx = 0
    env.controller.start_recording()

    while not done and step_idx < args.max_steps:
        # Tell the server which step the coming model call(s) belong to.
        client.step(episode_id, step_idx)
        response, actions = agent.predict(instruction, obs)
        for action in actions:
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            logger.info("Step %d: %s", step_idx + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)
            with open(os.path.join(result_dir,
                                   f"step_{step_idx + 1}_{action_timestamp}.png"), "wb") as f:
                f.write(obs["screenshot"])
            with open(os.path.join(result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                }) + "\n")
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1

    stop_reason = "complete" if done else "max_steps"
    try:
        env.controller.end_recording(os.path.join(result_dir, "recording.mp4"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("end_recording failed: %s", exc)
    return stop_reason


def build_agent(args):
    """Construct the unmodified agent selected by ``--agent``.

    Neither agent is given an explicit base URL: both fall back to
    ``OPENAI_BASE_URL`` on every call, which is how the worker routes each
    episode's completions to its episode-scoped server URL.
    """
    if args.agent == "holo3":
        return Holo3Agent(
            model=args.model,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            action_space=args.action_space,
            observation_type=args.observation_type,
            history_length=args.history_length,
            image_budget=args.keep_n_observations,
            enable_thinking=not args.disable_thinking,
            invert_scroll=args.invert_scroll,
            max_parse_retries=args.max_parse_retries,
            retry_temperature=args.retry_temperature,
        )
    if args.agent == "qwen":
        return QwenAgent(
            model=args.model,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            action_space=args.action_space,
            observation_type=args.observation_type,
            coordinate_type=args.coord,
            add_thought_prefix=args.add_thought_prefix,
            history_n=args.history_n,
            image_max=args.image_max,
            fold_size=args.fold_size,
            enable_thinking=args.enable_thinking,
        )
    raise ValueError(f"unknown agent {args.agent!r}")


def _worker_signal_handler(signum, frame):
    """In a worker, turn the first signal into an exception so the VM is closed.

    Raising unwinds to `run_env_worker`'s ``finally``, which powers the VM down;
    a second signal takes the default action, so a worker wedged in shutdown can
    still be killed from the keyboard.
    """
    signal.signal(signum, signal.SIG_DFL)
    raise KeyboardInterrupt(f"worker received signal {signum}")


def run_env_worker(task_queue, args) -> None:
    """One worker: stand up an env + agent, then run queued tasks as episodes."""
    # The fork carried the parent's handlers and its list of child handles in;
    # both are wrong here — the parent's handler would call `is_alive()` on
    # processes this one is not the parent of. Replace them before anything else.
    global processes
    processes = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _worker_signal_handler)

    client = ProfilingClient(args.server_url, args.model_path, args.runs_root,
                             start_timeout=args.start_timeout)
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

    env = None
    try:
        env = build_env(args)
        agent = build_agent(args)
        logger.info("Process %s started with %s.", current_process().name,
                    type(agent).__name__)

        consecutive_start_failures = 0
        consecutive_episode_failures = 0
        while True:
            try:
                domain, example_id = task_queue.get(timeout=5)
            except Exception:
                break

            # A fresh episode id per task, unique across workers, carried in the
            # agent's base URL so every completion is attributed to this rollout.
            episode_id = f"{domain}.{example_id}.{uuid.uuid4().hex[:8]}"
            os.environ["OPENAI_BASE_URL"] = client.episode_base_url(episode_id)

            result_dir = os.path.join(args.result_dir, domain, episode_id)
            os.makedirs(result_dir, exist_ok=True)
            with _episode_log(result_dir):
                config_file = os.path.join(
                    args.test_config_base_dir, f"examples/{domain}/{example_id}.json")
                with open(config_file, "r", encoding="utf-8") as f:
                    example = json.load(f)

                logger.info("[%s] %s/%s: %s", current_process().name, domain,
                            example_id, example.get("instruction", ""))
                try:
                    client.start(episode_id, task_id=example.get("id", example_id),
                                 domain=domain)
                except Exception as exc:  # noqa: BLE001
                    # One task lost, not the worker and its VM: the server may be
                    # loading a checkpoint, or busy, and the next task may well land.
                    consecutive_start_failures += 1
                    logger.error("Failed to start episode %s (%d in a row): %s",
                                 episode_id, consecutive_start_failures, exc)
                    if consecutive_start_failures >= MAX_CONSECUTIVE_START_FAILURES:
                        logger.error("Giving up: %d starts in a row failed.",
                                     consecutive_start_failures)
                        break
                    time.sleep(5)
                    continue
                consecutive_start_failures = 0
                stop_reason = "error"
                try:
                    stop_reason = run_episode(
                        client, episode_id, env, agent, example, domain, args, result_dir)
                    consecutive_episode_failures = 0
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    consecutive_episode_failures += 1
                    logger.error("Episode %s failed (%d in a row): %s", episode_id,
                                 consecutive_episode_failures, exc)
                    logger.error(traceback.format_exc())
                    if consecutive_episode_failures >= MAX_CONSECUTIVE_EPISODE_FAILURES:
                        # Another worker may still have a healthy env for it.
                        task_queue.put((domain, example_id))
                        logger.error("Giving up: %d episodes in a row failed; "
                                     "returned %s/%s to the queue.",
                                     consecutive_episode_failures, domain, example_id)
                        break
                finally:
                    try:
                        client.end(episode_id, stop_reason, score=None)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Failed to close episode %s: %s", episode_id, exc)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001
                logger.error("env close failed: %s", exc)


def config() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CUA-OP on-policy environment runner")

    # Where the profiling server is.
    p.add_argument("--server_url", required=True,
                   help="profiling server root, e.g. http://gpu-box:8000")
    # What the server profiles and where it writes: both are paths on the server.
    # Switching checkpoints between runs is fine; the server refuses a switch
    # while another runner still has episodes open on a different checkpoint.
    p.add_argument("--model_path", required=True,
                   help="checkpoint on the server to profile (loaded on demand)")
    p.add_argument("--runs_root", required=True,
                   help="directory on the server the episodes' runs go under")
    p.add_argument("--start_timeout", type=float, default=1800.0,
                   help="seconds to wait on /episode/start, which may load the checkpoint")

    # Environment / provider.
    add_env_args(p)
    p.add_argument("--sleep_after_execution", type=float, default=0.0)
    p.add_argument("--max_steps", type=int, default=15)

    # Agent (shared knobs).
    p.add_argument("--agent", type=str, default="holo3", choices=["holo3", "qwen"],
                   help="which vendored agent drives the episodes")
    p.add_argument("--model", type=str, default="onpolicy")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_tokens", type=int, default=4096)

    # Agent (holo3 only). The defaults are CUAPruning's, which is the point:
    # the agent loop is profiled on the inputs it was pruned under.
    p.add_argument("--history_length", type=int, default=6,
                   help="prior agent-loop steps replayed into the prompt "
                        "(CUAPruning's HISTORY_LENGTH)")
    p.add_argument("--keep_n_observations", type=int, default=3,
                   help="most-recent screenshots kept as images; older "
                        "<observation>s keep their wrapper but their pixels "
                        "become a placeholder (CUAPruning's IMAGE_BUDGET)")
    p.add_argument("--disable_thinking", action="store_true",
                   help="turn off holo3's reasoning channel. On by default: the "
                        "agent loop was profiled and pruned with thinking on")
    p.add_argument("--invert_scroll", action="store_true")
    p.add_argument("--max_parse_retries", type=int, default=3)
    p.add_argument("--retry_temperature", type=float, default=0.7)

    # Agent (qwen only; defaults mirror OSWorld's multi-env qwen runner).
    p.add_argument("--coord", type=str, default="relative",
                   choices=["absolute", "relative"],
                   help="coordinate system of the model's actions")
    p.add_argument("--add_thought_prefix", action="store_true")
    p.add_argument("--history_n", type=int, default=100)
    p.add_argument("--image_max", type=int, default=20)
    p.add_argument("--fold_size", type=int, default=10)
    p.add_argument("--enable_thinking", action="store_true",
                   help="send enable_thinking in extra_body (only honoured by DashScope URLs)")

    # Tasks.
    p.add_argument("--test_config_base_dir", type=str, default="evaluation_examples")
    p.add_argument("--test_all_meta_path", type=str,
                   default="evaluation_examples/test_all.json")
    p.add_argument("--domains", type=str, default="",
                   help="comma-separated subset of domains; empty means all")
    p.add_argument("--result_dir", type=str, default="./results_onpolicy")
    p.add_argument("--num_envs", type=int, default=1)

    # Preflight: validate the tasks and fill the download cache before any VM.
    p.add_argument("--skip_preflight", action="store_true",
                   help="start without checking tasks; downloads then happen mid-run")
    p.add_argument("--drop_failed_tasks", action="store_true",
                   help="run the tasks that pass preflight instead of refusing to start")

    p.add_argument("--log_level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return p.parse_args()


def _setup_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    root.addHandler(handler)


def _signal_handler(signum, frame):
    """Parent-side shutdown: terminate the workers, then wait for their VMs.

    Ctrl-C reaches every process in the group, and the parent's own
    `terminate()` sends a second signal on top, so this runs at most once and
    only in the process that owns the worker handles.
    """
    global _shutting_down
    if os.getpid() != _MAIN_PID or _shutting_down:
        return
    _shutting_down = True

    logger.info("Received signal %s; terminating workers.", signum)
    for proc in processes:
        try:
            if proc.is_alive():
                proc.terminate()
        except (AssertionError, ValueError, OSError) as exc:  # noqa: PERF203
            logger.warning("could not terminate %s: %s", proc.name, exc)

    # Powering a VM down takes a while; give the workers that time before
    # killing them, or the VMs are left running.
    deadline = time.time() + SHUTDOWN_GRACE
    for proc in processes:
        try:
            proc.join(timeout=max(0.0, deadline - time.time()))
            if proc.is_alive():
                logger.warning("%s did not exit in %.0fs; killing it.",
                               proc.name, SHUTDOWN_GRACE)
                proc.kill()
        except (AssertionError, ValueError, OSError) as exc:  # noqa: PERF203
            logger.warning("could not reap %s: %s", proc.name, exc)
    sys.exit(0)


def main() -> int:
    args = config()
    _setup_logging(args.log_level)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.provider_name == "vmware":
        vm_dir = (os.path.dirname(os.path.abspath(args.path_to_vm)) if args.path_to_vm
                  else "vmware_vm_data")
        free = shutil.disk_usage(vm_dir if os.path.isdir(vm_dir) else ".").free
        needed = args.num_envs * VMWARE_FREE_BYTES_PER_ENV
        if free < needed:
            logger.error("Only %.1f GiB free for the VMs; %d envs need about %.1f GiB.",
                         free / 2**30, args.num_envs, needed / 2**30)
            return 1

    tasks = select_tasks(args.test_all_meta_path, args.domains)
    if not args.skip_preflight:
        _, report = run_preflight(args.test_config_base_dir, tasks)
        if report.errors:
            if not args.drop_failed_tasks:
                logger.error("Preflight failed for %d task(s); fix them, or pass "
                             "--drop_failed_tasks to run the rest.", len(report.errors))
                return 1
            tasks = [t for t in tasks if f"{t[0]}/{t[1]}" not in report.errors]
            logger.warning("Dropped %d task(s) that failed preflight.", len(report.errors))
        # Every download is cached now: a miss mid-run is a bug, not a reason to
        # wait on the network. The workers inherit this.
        os.environ[OFFLINE_ENV_VAR] = "1"
    logger.info("Total tasks: %d across %d domains", len(tasks),
                len({domain for domain, _ in tasks}))

    with Manager() as manager:
        task_queue = manager.Queue()
        for task in tasks:
            task_queue.put(task)
        global processes
        processes = []
        for i in range(args.num_envs):
            proc = Process(target=run_env_worker, args=(task_queue, args),
                           name=f"EnvProcess-{i + 1}", daemon=True)
            proc.start()
            processes.append(proc)
            logger.info("Started %s (pid %s)", proc.name, proc.pid)
        for proc in processes:
            proc.join()
    logger.info("All episodes finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
