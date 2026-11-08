"""Read a saved profiling run, one generation at a time.

`Run.open` takes a run's run.meta.json; `Run.generations` yields each generation's
stats, and `Run.outputs` its text and token counts.
"""
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from .store import read_generation, read_outputs


def _stored_layout(path: Path) -> tuple[int, int]:
    """(k, E) as one generation's ``.npz`` records them: k is the width of
    ``norms``, one column per selected expert, and E is stored outright."""
    with np.load(path) as z:
        return int(z["norms"].shape[-1]), int(z["n_experts"])


@dataclass
class Run:
    """One profiling run, resolved to the files an analysis pass needs."""

    stem: Path
    """``<dir>/<name>``, the run's shared prefix; ``.meta.json`` etc. hang off it."""
    meta: dict
    """The run's ``*.meta.json``, or ``{}`` when it could not be found."""
    top_k: int
    """Experts per token. The stored arrays keep >= k columns; k is the selection."""
    n_experts: int
    """E, the experts each router chooses from."""
    stats_files: list[Path] = field(default_factory=list)
    """The per-generation ``gen_*.npz``, in order."""
    outputs_path: Path | None = None
    """The ``*.outputs.jsonl`` text sidecar, or None when there is none."""

    @classmethod
    def open(cls, path: Path, top_k: int | None = None,
             n_experts: int | None = None, outputs: Path | None = None) -> "Run":
        """
        Resolve a run from whatever part of it the caller happens to name.

        Args:
            path: The run's ``*.meta.json`` or its stats directory.
            top_k: Experts per token; default: the meta's, else what the
                stats record. Below the recorded k, it narrows the selection.
            n_experts: E; default: the meta's, else what the stats record.
            outputs: The ``*.outputs.jsonl``; default: found next to the run.

        Returns:
            A `Run` whose stats files have been checked to exist.

        Raises:
            FileNotFoundError: No ``gen_*.npz`` were found for the run.
        """
        path = Path(path)
        stem, meta, stats_files = cls._resolve(path)
        if not stats_files:
            raise FileNotFoundError(
                f"no routing statistics under {path}: expected gen_*.npz")
        meta = meta or cls._find_meta(stem)

        top_k = top_k or meta.get("top_k")
        n_experts = n_experts or meta.get("n_experts")
        if not (top_k and n_experts):
            stored_k, stored_e = _stored_layout(stats_files[0])
            top_k, n_experts = top_k or stored_k, n_experts or stored_e

        outputs = Path(outputs) if outputs else stem.parent / f"{stem.name}.outputs.jsonl"
        return cls(stem=stem, meta=meta, top_k=int(top_k), n_experts=int(n_experts),
                   stats_files=stats_files,
                   outputs_path=outputs if outputs.exists() else None)

    @staticmethod
    def _resolve(path: Path) -> tuple[Path, dict, list[Path]]:
        """
        Resolve a run from whatever part of it the caller happens to name.

        Args:
            path: The run's meta file or its stats directory.

        Returns:
            A tuple of the run's stem, meta, and stats files.
        """
        if path.is_dir():
            return path.parent / path.name.removesuffix("_stats"), {}, sorted(path.glob("gen_*.npz"))

        meta = json.loads(path.read_text())
        stem = path.with_suffix("")
        if stem.suffix == ".meta":                    # <name>.meta.json
            stem = stem.with_suffix("")
        candidates = [Path(meta["stats_dir"])] if meta.get("stats_dir") else []
        candidates.append(stem.parent / f"{stem.name}_stats")
        for cand in candidates:
            if cand.is_dir():
                found = sorted(cand.glob("gen_*.npz"))
                if found:
                    return stem, meta, found
        return stem, meta, []

    @staticmethod
    def _find_meta(stem: Path) -> dict:
        """The run's ``*.meta.json`` when the caller named the stats instead."""
        cand = stem.parent / f"{stem.name}.meta.json"
        return json.loads(cand.read_text()) if cand.exists() else {}

    @cached_property
    def outputs(self) -> list[dict]:
        """
        One text record per generation, in the same order as `stats_files`,
        read on first access. Empty when the run has no outputs sidecar.

        Raises:
            ValueError: The sidecar's rows do not line up one to one with the
                stats files.
        """
        if self.outputs_path is None:
            return []
        rows = read_outputs(self.outputs_path)
        if len(rows) != self.n_generations:
            raise ValueError(f"{self.n_generations} stats files but {len(rows)} "
                             f"output rows in {self.outputs_path}; they must "
                             f"correspond one to one")
        return rows

    @property
    def n_generations(self) -> int:
        return len(self.stats_files)

    def generations(self, indices=None) -> Iterator[tuple[int, dict]]:
        """
        Iterate the run one generation at a time, yielding the generation index and its stats record.

        Args:
            indices: Optional set of generation indices to yield. If None, yield all generations.

        Yields:
            A tuple of the generation index and its corresponding stats record.
        """
        wanted = None if indices is None else {int(i) for i in indices}
        for gi, path in enumerate(self.stats_files):
            if wanted is None or gi in wanted:
                yield gi, read_generation(path, self.top_k)
