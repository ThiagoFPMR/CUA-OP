"""The profiling server: one hooked MoE model behind an OpenAI-compatible API.

    POST /episode/start        {episode_id, model_path, runs_root, task_id?, domain?}
    POST /ep/{id}/step         {step, stage?}
    POST /ep/{id}/v1/chat/completions
    POST /ep/{id}/end          {stop_reason, score?}
    GET  /status

Generations run one at a time, each written to its episode's run (see `episodes`).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import anyio
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from ..loading import load_model_and_processor, text_config
from ..profiler import MoEProfiler
from ..store import DEFAULT_TOP_M

from .decode_graph import DecodeGraph
from .episodes import EpisodeClosed, EpisodeRegistry
from .generation import (GenerationAbandoned, PromptTooLarge, openai_to_prompt,
                         profile_generation, register_probes,
                         release_cuda_memory)

logger = logging.getLogger("moe_profiler.server")


@dataclass
class ServerConfig:
    """How the server loads and profiles whichever checkpoint a client names."""

    attn: str = "sdpa"
    moe_scope: str | None = None
    top_k: int | None = None
    top_m: int = DEFAULT_TOP_M
    sample: bool = False
    verify: bool = False
    # The model lock is a queue nothing else drains, and a client that gives up
    # and retries only lengthens it. Past either bound the server says so —
    # a 503 a client can back off from beats a request that never returns.
    max_queued: int = 16
    generate_timeout: float = 1800.0
    # Longest prompt, in input ids with image tokens included, the server will
    # generate from. None leaves it to the GPU — an out-of-memory error is then
    # recovered from, but only after the forward has run into it.
    max_prompt_tokens: int | None = None
    # Generations never overlap, so between two of them nothing is using the
    # GPU and handing cached memory back costs only the next request's
    # re-allocation. Every this many generations; 0 disables it.
    release_every: int = 1
    # Replay each decode step from a CUDA graph (see decode_graph): one launch
    # per token instead of thousands. Needs a static cache, whose attention
    # kernels differ from the default cache's, so outputs can differ from an
    # eager run at the level of bf16 rounding.
    decode_graphs: bool = True


# ------------------------------------------------------------------ requests

class StartRequest(BaseModel):
    episode_id: str
    # The checkpoint to profile (a path on the server, or anything
    # `from_pretrained` accepts) and the directory the episode's run goes under.
    model_path: str
    runs_root: str
    task_id: str | None = None
    domain: str | None = None


class StepRequest(BaseModel):
    step: int
    stage: str | None = None


class EndRequest(BaseModel):
    stop_reason: str = "complete"
    score: float | None = None


class ChatRequest(BaseModel):
    # Permissive: an agent may send fields beyond the ones acted on here.
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[dict]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    # An agent passes these via the OpenAI client's extra_body, which arrives at
    # the top level of the request; they reach the chat template unchanged.
    chat_template_kwargs: dict | None = None


# ------------------------------------------------------------------- server

def build_profiler(config: ServerConfig, model_path: str) -> MoEProfiler:
    """Load a checkpoint and build its probes, with the routing
    hyperparameters read from the model config."""
    model, processor = load_model_and_processor(model_path, config.attn)
    _log_placement(model)
    cfg = text_config(model)
    top_k = config.top_k or getattr(cfg, "num_experts_per_tok", None) \
        or getattr(cfg, "num_selected_experts", 8)
    norm_topk_prob = bool(getattr(cfg, "norm_topk_prob", True))
    scoring_func = getattr(cfg, "scoring_func", "softmax")
    return MoEProfiler(
        model, processor, top_k, scoring_func, norm_topk_prob,
        config.moe_scope, config.verify,
        {"model_path": model_path,
         "model_type": getattr(model.config, "model_type", None)},
        do_sample=config.sample, top_m=config.top_m)


def _log_placement(model) -> None:
    """Log what a checkpoint landed on, so timings compare across machines:
    the GPUs and host CPU, how its modules spread over the GPUs (any on ``cpu``
    or ``disk`` are offloaded), the attention and experts paths, whether the
    fast linear-attention kernels are installed, and the allocator settings."""
    import importlib.util
    import platform

    import torch
    import transformers
    gpus = [f"{i}: {torch.cuda.get_device_name(i)} "
            f"{torch.cuda.get_device_properties(i).total_memory / 2**30:.0f} GiB"
            for i in range(torch.cuda.device_count())]
    # Decoding one token at a time is bound by how fast one host core launches
    # kernels, so the CPU model matters as much as the GPU.
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next((line.split(":", 1)[1].strip() for line in f
                        if line.startswith("model name")), cpu)
    except OSError:
        pass
    spread = Counter(str(d) for d in (getattr(model, "hf_device_map", None) or {}).values())
    cfg = text_config(model)
    logger.info("placement: gpus=%s cpu=%s x%s modules_per_device=%s attn=%s "
                "experts=%s fla=%s torch=%s transformers=%s "
                "PYTORCH_CUDA_ALLOC_CONF=%s",
                gpus or "none", cpu, os.cpu_count(), dict(spread),
                getattr(model.config, "_attn_implementation", None),
                getattr(cfg, "_experts_implementation", None),
                importlib.util.find_spec("fla") is not None,
                torch.__version__, transformers.__version__,
                os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))


# The phases a chat request's wall time is split into, in the order it passes
# through them. Whatever they do not cover is reported as "other".
_PHASES = ("parse_s", "wait_s", "encode_s", "prefill_s", "decode_s", "pack_s",
           "release_s", "record_s")


def _timing_line(episode_id: str, step: int, outcome: str, t: dict) -> str:
    """One chat request's wall time, phase by phase, for the server log.

    parse: body upload, JSON and validation. wait: the queue for the model.
    prefill/decode: `generate`, with the host time the probe hooks took to
    launch their work in each (their GPU time overlaps the model's and is not
    split out). pack: finalizing and packing the stats. release: freeing GPU
    memory, under the model lock. record: writing the run's files, after the
    lock. A request that failed shows the phases it got through; the rest read 0.
    """
    g = lambda k: t.get(k, 0.0)  # noqa: E731
    steps = t.get("decode_steps", 0)
    other = g("total_s") - sum(g(k) for k in _PHASES)
    return (f"timing {episode_id} step {step} [{outcome}]: "
            f"total {g('total_s'):.1f}s = "
            f"parse {g('parse_s'):.1f} + wait {g('wait_s'):.1f} + "
            f"encode {g('encode_s'):.1f} + "
            f"prefill {g('prefill_s'):.1f} (probe host {g('prefill_probe_host_s'):.1f}) + "
            f"decode {g('decode_s'):.1f} (probe host {g('decode_probe_host_s'):.1f}; "
            f"{steps} steps, {1000 * g('decode_s') / max(steps, 1):.0f} ms/step) + "
            f"pack {g('pack_s'):.1f} + release {g('release_s'):.1f} + "
            f"record {g('record_s'):.1f} + other {other:.1f}; "
            f"generate thread cpu {g('generate_cpu_s'):.1f}s "
            f"of {g('generate_s'):.1f}s"
            + (f"; decode steps replayed from a CUDA graph "
               f"{t['graph_replays']}, eager {t['graph_eager_steps']}, "
               f"cache grown {t.get('graph_growths', 0)}x"
               if "graph_replays" in t else "; decode eager"))


class _ArrivalStamp:
    """ASGI middleware: stamp each request with when its headers arrived.

    A handler runs only once its body — for a chat request, several base64
    screenshots — is uploaded, parsed and validated; the stamp lets it count
    that time too.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope.setdefault("state", {})["arrived"] = time.perf_counter()
        await self.app(scope, receive, send)


class CheckpointBusy(RuntimeError):
    """A different checkpoint was requested while episodes are open on this one."""


class GenerateBusy(RuntimeError):
    """The generation queue is full, or the wait for the model ran out."""


class FifoLock:
    """A lock its waiters get in the order they asked for it.

    ``threading.Lock`` hands itself to whichever waiter the OS wakes first, and
    a thread arriving just as it is released can take it before any of them.
    With generations queued back to back, one environment's request can lose
    that race over and over while the others each get several turns -- a wait
    of many generations where the queue was only a few deep.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._held = False
        self._line: deque = deque()

    def acquire(self, timeout: float | None = None,
                give_up: Callable[[], bool] | None = None, poll: float = 1.0) -> bool:
        """Wait for the lock, keeping this caller's place in line. Leave the
        line and return False after ``timeout`` seconds, or as soon as
        ``give_up`` says so (checked every ``poll`` seconds, under the lock's
        own condition, so it must be quick)."""
        me, acquired = object(), False
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            self._line.append(me)
            try:
                while self._held or self._line[0] is not me:
                    wait = None if give_up is None else poll
                    if deadline is not None:
                        left = deadline - time.monotonic()
                        if left <= 0:
                            return False
                        wait = left if wait is None else min(wait, left)
                    if give_up is not None and give_up():
                        return False
                    self._cond.wait(wait)
                self._held = acquired = True
                return True
            finally:
                self._line.remove(me)
                if not acquired:            # gave up, or interrupted
                    self._cond.notify_all() # the next in line may be first now

    def release(self) -> None:
        with self._cond:
            if not self._held:
                raise RuntimeError("release of an unheld FifoLock")
            self._held = False
            self._cond.notify_all()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


class ProfilingServer:
    """Holds the resident model, its probes, the episode table, and the lock.

    ``loader`` builds a profiler for a checkpoint path; it defaults to loading
    the real model and may be replaced to exercise the lifecycle without one.
    """

    def __init__(self, config: ServerConfig,
                 loader: Callable[[str], MoEProfiler] | None = None):
        self.config = config
        self._loader = loader or (lambda path: build_profiler(config, path))
        # Two locks, always taken in this order where both are needed. The model
        # lock is held for a whole generation — minutes — so nothing that must
        # stay responsive may wait on it; the state lock guards which checkpoint
        # is resident and is never held across one. The model lock serves its
        # queue first come, first served (see FifoLock).
        self.model_lock = FifoLock()
        self.state_lock = threading.RLock()
        self.registry = EpisodeRegistry()
        self._queued = 0
        self._queue_lock = threading.Lock()
        self._since_release = 0

        self.model_path: str | None = None
        self.profiler: MoEProfiler | None = None
        self.run_meta_base: dict = {}
        self._handles: list = []
        self.decode_graph: DecodeGraph | None = None

    @property
    def queued(self) -> int:
        """Generations waiting for, or holding, the model lock."""
        with self._queue_lock:
            return self._queued

    def start_episode(self, req: StartRequest):
        """Make ``req.model_path`` resident, then register the episode on it.

        Both happen under the state lock, so no other start can slip an episode
        onto the outgoing checkpoint. Registering an episode on the resident
        checkpoint — every start but the first — touches the model lock not at
        all, so a runner resetting one environment is not made to wait out
        another environment's generation. Only a swap takes the model lock, and
        a swap happens only with no episode open.
        """
        with self.state_lock:
            if req.model_path != self.model_path:
                if len(self.registry):
                    raise CheckpointBusy(
                        f"{self.model_path!r} is resident with "
                        f"{len(self.registry)} open episode(s); end them before "
                        f"switching to {req.model_path!r}")
                self._swap(req.model_path)
            return self.registry.start(req.episode_id, req.task_id, req.domain,
                                       Path(req.runs_root), dict(self.run_meta_base))

    def _swap(self, model_path: str) -> None:
        """Unload the resident checkpoint (if any) and load ``model_path``.

        Under the model lock: no episode is open, but a generation abandoned by
        the runner that closed its episode may still be in flight, and it must
        not find the model pulled out from under it.
        """
        with self.model_lock:
            self._swap_locked(model_path)

    def _swap_locked(self, model_path: str) -> None:
        """The body of `_swap`, under the model lock."""
        if self.profiler is not None:
            logger.info("unloading %s", self.model_path)
            for handle in self._handles:
                handle.remove()
            if self.decode_graph is not None:
                self.decode_graph.uninstall()
            self._handles, self.profiler, self.model_path = [], None, None
            self.decode_graph = None
            self.run_meta_base = {}
            release_cuda_memory()

        logger.info("loading %s", model_path)
        profiler = self._loader(model_path)
        # Hook the MoE blocks for as long as this checkpoint stays resident.
        self._handles = register_probes(profiler)
        model = getattr(profiler, "model", None)
        if self.config.decode_graphs and model is not None \
                and next(model.parameters()).is_cuda:
            self.decode_graph = DecodeGraph(model, profiler.probes)
        self.profiler, self.model_path = profiler, model_path
        self.run_meta_base = {
            "n_moe_blocks": len(profiler.blocks),
            "layer_indices": profiler.layer_indices,
            "module_names": profiler.module_names,
            "top_k": profiler.top_k,
            "context_window": profiler.context_window,
            "do_sample": profiler.do_sample,
            "model_path": model_path,
            "model_type": profiler.meta_extra.get("model_type"),
            "decode_graphs": self.decode_graph is not None,
        }

    def generate(self, prompt: dict,
                 abandoned: Callable[[], bool] = lambda: False,
                 timings: dict | None = None):
        """Profile one generation, serialized against every other.

        The probes hold per-generation state, so generations cannot interleave
        and the model lock is the server's one real queue. Both its depth and the
        wait for it are bounded: past either, the caller is told to back off
        rather than holding a worker thread until its own timeout fires.

        ``abandoned`` reports that the caller has gone away. It is checked while
        waiting for the model, again once the model is ours, and on every
        decoded token, so an abandoned request gives up its place instead of
        spending the GPU on an answer no one will read.

        ``timings`` is filled phase by phase, the wait for the model and the
        memory release around the generation's own (see `profile_generation`).

        Raises:
            GenerateBusy: The queue is full, or the model did not come free in
                ``generate_timeout`` seconds.
            GenerationAbandoned: ``abandoned`` fired before the answer was done.
            PromptTooLarge: The prompt is over ``max_prompt_tokens``, or the
                GPU ran out of memory on it.
        """
        with self._queue_lock:
            if self._queued >= self.config.max_queued:
                raise GenerateBusy(
                    f"{self._queued} generations already queued "
                    f"(max_queued={self.config.max_queued})")
            self._queued += 1
        timings = {} if timings is None else timings
        try:
            queued_at = time.perf_counter()
            deadline = time.monotonic() + self.config.generate_timeout
            try:
                if not self.model_lock.acquire(
                        give_up=lambda: abandoned() or time.monotonic() >= deadline):
                    if abandoned():
                        raise GenerationAbandoned("the caller left while queued")
                    raise GenerateBusy(
                        f"waited {self.config.generate_timeout:g}s for the model")
            finally:
                timings["wait_s"] = time.perf_counter() - queued_at
            try:
                if abandoned():
                    raise GenerationAbandoned("the caller left while queued")
                return profile_generation(
                    self.profiler, prompt, should_stop=abandoned,
                    max_prompt_tokens=self.config.max_prompt_tokens,
                    decode_graph=self.decode_graph,
                    timings=timings)
            finally:
                releasing = time.perf_counter()
                self._release_between_generations()
                timings["release_s"] = time.perf_counter() - releasing
                self.model_lock.release()
        finally:
            with self._queue_lock:
                self._queued -= 1


    def _release_between_generations(self) -> None:
        """Every ``release_every`` generations, free what the last ones left.

        Called under the model lock, after a generation has returned or failed:
        no forward is running, so nothing on the device is live but the weights.
        This clears garbage cycles still pinning activations (an abandoned
        generation's traceback, say) and the allocator's cached segments, so the
        next long prompt starts from the weights alone, not a fragmented cache.
        """
        if not self.config.release_every:
            return
        self._since_release += 1
        if self._since_release >= self.config.release_every:
            self._since_release = 0
            release_cuda_memory()


def create_app(config: ServerConfig,
               loader: Callable[[str], MoEProfiler] | None = None) -> FastAPI:
    """Build the FastAPI app around a server that loads checkpoints on request.

    A ``loader`` may be injected in place of loading real checkpoints — used to
    exercise the request lifecycle without a model.
    """
    server = ProfilingServer(config, loader=loader)
    app = FastAPI(title="MoE profiling server")
    app.add_middleware(_ArrivalStamp)
    app.state.server = server

    @app.get("/status")
    def status():
        return {"model_path": server.model_path,
                "open_episodes": len(server.registry),
                "queued_generations": server.queued}

    @app.post("/episode/start")
    def episode_start(req: StartRequest):
        try:
            episode = server.start_episode(req)
        except (KeyError, CheckpointBusy) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {"episode_id": req.episode_id, "model_path": req.model_path,
                "run_dir": str(episode.dir)}

    @app.post("/ep/{episode_id}/step")
    def episode_step(episode_id: str, req: StepRequest):
        try:
            server.registry.get(episode_id).set_cursor(req.step, req.stage)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"episode_id": episode_id, "step": req.step, "stage": req.stage}

    async def _watch_disconnect(request: Request, gone: threading.Event):
        while not await request.is_disconnected():
            await asyncio.sleep(1.0)
        gone.set()

    @app.post("/ep/{episode_id}/v1/chat/completions")
    async def chat_completions(episode_id: str, req: ChatRequest,
                               request: Request):
        entered = time.perf_counter()
        arrived = getattr(request.state, "arrived", entered)
        try:
            episode = server.registry.get(episode_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

        prompt = openai_to_prompt(req.messages, max_new_tokens=req.max_tokens,
                                  chat_template_kwargs=req.chat_template_kwargs)
        # The generation runs on a worker thread while this handler watches the
        # connection: an agent whose HTTP client timed out has hung up, and its
        # request is dropped rather than left to generate for no one.
        gone = threading.Event()
        timings: dict = {}

        def _generate_and_record():
            # Recording happens after the model lock is released: the stats are
            # already in memory and the episode's files are its own, so it need
            # not block the model. The episode is claimed across both, so one
            # closed underneath a slow generation refuses the write.
            outcome = "ok"
            try:
                with episode.generating():
                    result = server.generate(prompt, abandoned=gone.is_set,
                                             timings=timings)
                    recording = time.perf_counter()
                    episode.record(result)
                    timings["record_s"] = time.perf_counter() - recording
                return result
            except BaseException as exc:
                outcome = type(exc).__name__
                raise
            finally:
                # Every outcome, not only success: the slowest generations are
                # the ones a client gives up on.
                timings.update(parse_s=entered - arrived,
                               total_s=time.perf_counter() - arrived)
                logger.info("%s", _timing_line(episode_id, episode.step,
                                               outcome, timings))

        watcher = asyncio.create_task(_watch_disconnect(request, gone))
        try:
            result = await anyio.to_thread.run_sync(_generate_and_record)
        except GenerationAbandoned as exc:
            logger.warning("dropping a generation for %s: %s", episode_id, exc)
            raise HTTPException(status_code=499, detail=str(exc))
        except EpisodeClosed as exc:
            # The runner gave up on this rollout while the request was queued or
            # generating, and has already closed the run's files. Say so, so the
            # agent stops retrying into a rollout that no longer exists.
            logger.warning("dropping a generation for %s: %s", episode_id, exc)
            raise HTTPException(status_code=409, detail=str(exc))
        except GenerateBusy as exc:
            logger.warning("refusing a generation for %s: %s", episode_id, exc)
            raise HTTPException(status_code=503, detail=str(exc))
        except PromptTooLarge as exc:
            # Not a 5xx: the same prompt would fail the same way, so the agent
            # should not retry it, and the server is healthy for the next one.
            logger.warning("refusing a generation for %s: %s", episode_id, exc)
            raise HTTPException(status_code=413, detail=str(exc))
        finally:
            watcher.cancel()

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or episode.run_meta.get("model_path"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.generated_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": result.n_prompt_input_ids,
                "completion_tokens": len(result.generated_token_ids),
                "total_tokens": result.n_prompt_input_ids
                + len(result.generated_token_ids),
            },
        }

    @app.post("/ep/{episode_id}/end")
    def episode_end(episode_id: str, req: EndRequest):
        try:
            meta_path = server.registry.end(episode_id, req.stop_reason, req.score)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"episode_id": episode_id, "meta_path": str(meta_path)}

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description="MoE routing profiling server")
    # No checkpoint or runs root here: each /episode/start names both.
    ap.add_argument("--host", default=os.environ.get("MOE_PROFILER_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("MOE_PROFILER_PORT", "8000")))
    ap.add_argument("--attn", default=os.environ.get("ATTN_IMPL", "sdpa"))
    ap.add_argument("--moe-scope", default=os.environ.get("MOE_SCOPE"))
    ap.add_argument("--top-k", type=int,
                    default=int(os.environ["TOP_K"]) if os.environ.get("TOP_K") else None)
    ap.add_argument("--top-m", type=int,
                    default=int(os.environ.get("TOP_M", DEFAULT_TOP_M)))
    ap.add_argument("--max-queued", type=int,
                    default=int(os.environ.get("MAX_QUEUED", "16")),
                    help="generations that may wait on the model before 503s start")
    ap.add_argument("--generate-timeout", type=float,
                    default=float(os.environ.get("GENERATE_TIMEOUT", "1800")),
                    help="seconds a queued generation waits for the model before 503")
    ap.add_argument("--max-prompt-tokens", type=int,
                    default=int(os.environ["MAX_PROMPT_TOKENS"])
                    if os.environ.get("MAX_PROMPT_TOKENS") else None,
                    help="refuse (413) prompts longer than this many input ids, "
                         "image tokens included; the log's per-generation peak "
                         "memory shows where the GPU's limit falls")
    ap.add_argument("--release-every", type=int,
                    default=int(os.environ.get("RELEASE_EVERY", "1")),
                    help="free cached GPU memory every N generations (0: never)")
    ap.add_argument("--sample", action="store_true",
                    help="sample instead of greedy (greedy keeps routing reproducible)")
    ap.add_argument("--verify", action="store_true",
                    help="run the one-off fused-reconstruction cross-check")
    ap.add_argument("--no-decode-graphs", action="store_true",
                    default=os.environ.get("DECODE_GRAPHS", "1") == "0",
                    help="decode eagerly instead of replaying each step from a "
                         "CUDA graph (slower; the default cache's kernels)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    import uvicorn
    config = ServerConfig(
        attn=args.attn, moe_scope=args.moe_scope, top_k=args.top_k,
        top_m=args.top_m, sample=args.sample, verify=args.verify,
        max_queued=args.max_queued, generate_timeout=args.generate_timeout,
        max_prompt_tokens=args.max_prompt_tokens,
        release_every=args.release_every,
        decode_graphs=not args.no_decode_graphs)
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
