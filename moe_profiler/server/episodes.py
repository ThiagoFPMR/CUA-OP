"""Open episodes, and the run each one writes to disk.

Each episode writes `<runs_root>/<episode>/`: run.meta.json, run.outputs.jsonl, and
one run_stats/gen_*.npz per generation, which `moe_profiler.run.Run` reads back.
"""
from __future__ import annotations

import contextlib
import json
import threading
from pathlib import Path

from ..store import FORMAT_VERSION, write_generation

from .generation import GenerationResult


class EpisodeClosed(RuntimeError):
    """A generation arrived for an episode that has already been closed.

    The runner closes an episode as soon as its rollout ends, including when it
    ended because the agent gave up waiting on a queued request. That request may
    still be generating, and lands here when it finishes.
    """


def _safe_id(episode_id: str) -> str:
    """A filesystem-safe directory name for an episode id from a URL segment."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in episode_id)


class Episode:
    """One rollout's accumulating run: its output files, counters, and cursor.

    The cursor — the current step and stage — is set by the runner before each
    agent decision and stamped onto every generation recorded until it moves.

    Recording is deliberately not serialized by the server's model lock, so the
    episode guards its own files: `record` and `close` contend for one lock, and
    a generation that outlives the episode it belongs to is refused rather than
    written to a file that has since been closed.
    """

    def __init__(self, episode_id: str, task_id: str | None,
                 domain: str | None, root: Path, run_meta: dict | None = None):
        self.id = episode_id
        self.task_id = task_id
        self.domain = domain
        # Model-level metadata of the checkpoint this episode runs against,
        # fixed at start: the server may swap checkpoints between episodes.
        self.run_meta = run_meta or {}
        self.dir = root / _safe_id(episode_id)
        self.stats_dir = self.dir / "run_stats"
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        self._stem = self.dir / "run"
        self._out_fp = self._stem.with_suffix(".outputs.jsonl").open("w")
        # Held across a whole record, and by close; `_closed` is the handoff
        # between them, so neither ever sees the other's half-done work.
        self._lock = threading.Lock()
        self._closed = False

        self.step = 0
        self.stage: str | None = None
        self._n_generations = 0
        self._prev_prompt_ids: list[int] = []
        self._layout: dict | None = None

    def set_cursor(self, step: int, stage: str | None) -> None:
        """Point the episode at the step (and optional stage) now being run."""
        self.step = step
        self.stage = stage

    @contextlib.contextmanager
    def generating(self):
        """Claim this episode for one in-flight generation.

        Entered before the model runs, so a request that queued behind others
        while its rollout was abandoned is turned away before it spends the GPU,
        rather than only at `record`.
        """
        with self._lock:
            if self._closed:
                raise EpisodeClosed(f"episode {self.id!r} has already ended")
        yield

    def record(self, result: GenerationResult) -> None:
        """
        Write one generation's routing stats and its outputs row.

        The stats go to ``gen_<i>.npz``; the row carries the text, the token
        counts that align it to the stats, and the current step/stage/domain. The
        prompt's ids are stored as the length shared with the previous row's
        prompt plus the tokens past it, since consecutive steps repeat a long
        prefix.

        Raises:
            EpisodeClosed: The episode ended while this generation was still in
                flight; nothing is written.
        """
        with self._lock:
            if self._closed:
                raise EpisodeClosed(
                    f"episode {self.id!r} ended while this generation was in flight")
            self._record(result)

    def _record(self, result: GenerationResult) -> None:
        """The body of `record`, under the episode's lock."""
        i = self._n_generations
        write_generation(self.stats_dir / f"gen_{i:05d}.npz", result.arrays)
        self._layout = result.layout

        prompt_ids = result.prompt_token_ids
        shared = 0
        for a, b in zip(self._prev_prompt_ids, prompt_ids):
            if a != b:
                break
            shared += 1
        self._prev_prompt_ids = prompt_ids

        self._out_fp.write(json.dumps({
            "step": self.step,
            "stage": self.stage,
            "domain": self.domain,
            "generated_text": result.generated_text,
            "generated_token_ids": result.generated_token_ids,
            "prompt_text": result.prompt_text,
            "prompt_shared_prefix": shared,
            "prompt_token_ids_suffix": prompt_ids[shared:],
            "n_prompt_input_ids": result.n_prompt_input_ids,
            "n_prefill_tokens": result.n_prefill_tokens,
            # Rows [0, n_prefill) are the prompt; rows [n_prefill, n_activation_rows)
            # are the generated tokens fed back in, so row n_prefill + j is
            # generated_token_ids[j]. The last generated token is never fed back.
            "n_generated_tokens": len(result.generated_token_ids),
            "n_activation_rows": result.n_activation_rows,
            # Served by the retry after an out-of-memory error: same forward,
            # images encoded one at a time and the attention cache on the CPU.
            "low_memory": result.low_memory,
            # Seconds per phase, as far as they are known before this write:
            # the model's phases and the queue, not the upload or the write.
            "timings": {k: round(v, 3) for k, v in result.timings.items()},
        }) + "\n")
        self._out_fp.flush()
        self._n_generations += 1

    def close(self, stop_reason: str, score) -> Path:
        """
        Finalize the episode: write ``run.meta.json`` and close the outputs file.

        Args:
            stop_reason: Why the rollout ended (``complete`` / ``max_steps`` /
                ``unparsed`` / ``error``).
            score: An optional success label; ``None`` for tasks without an
                evaluator.

        Returns:
            The path to the written ``run.meta.json``.
        """
        with self._lock:
            self._closed = True
            return self._close(stop_reason, score)

    def _close(self, stop_reason: str, score) -> Path:
        """The body of `close`, under the episode's lock."""
        layout = self._layout or {}
        meta = {
            **self.run_meta,
            "episode_id": self.id,
            "task_id": self.task_id,
            "domain": self.domain,
            "stop_reason": stop_reason,
            "score": score,
            "generations_written": self._n_generations,
            "steps_written": self.step + 1 if self._n_generations else 0,
            "n_experts": layout.get("n_experts"),
            "top_k": layout.get("top_k", self.run_meta.get("top_k")),
            "expert_layout": layout.get("expert_layout"),
            "scores_scattered": layout.get("scores_scattered"),
            "stats_top_m": layout.get("stats_top_m"),
            "stats_dir": str(self.stats_dir),
            "outputs_path": str(self._stem.with_suffix(".outputs.jsonl")),
            "stats_format_version": FORMAT_VERSION,
        }
        meta_path = self._stem.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))
        self._out_fp.close()
        return meta_path


class EpisodeRegistry:
    """The live episodes, keyed by the id carried in the request URL.

    Each episode names its own runs root, so one server can write different
    checkpoints' runs to different places.
    """

    def __init__(self):
        self._episodes: dict[str, Episode] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._episodes)

    def start(self, episode_id: str, task_id: str | None, domain: str | None,
              root: Path, run_meta: dict | None = None) -> Episode:
        """Register a new episode and open its output directory under ``root``."""
        with self._lock:
            if episode_id in self._episodes:
                raise KeyError(f"episode {episode_id!r} is already open")
            episode = Episode(episode_id, task_id, domain, Path(root), run_meta)
            self._episodes[episode_id] = episode
            return episode

    def get(self, episode_id: str) -> Episode:
        """The open episode for this id, or KeyError if it was never started."""
        with self._lock:
            try:
                return self._episodes[episode_id]
            except KeyError:
                raise KeyError(f"unknown episode {episode_id!r}; start it first")

    def end(self, episode_id: str, stop_reason: str, score) -> Path:
        """Close an episode, write its metadata, and drop it from the table."""
        with self._lock:
            episode = self._episodes.pop(episode_id)
        return episode.close(stop_reason, score)
