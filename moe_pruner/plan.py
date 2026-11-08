"""PruningPlan: which experts each layer keeps, saved as readable JSON.

A plan lists the kept expert ids of each layer (original ids, ascending), the
model layout it applies to, how the budget was set, and which runs it came from.
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np

from .budget import Allocation, uniform_curve

FORMAT_VERSION = 1


@dataclass
class PruningPlan:
    """Which routed experts survive, per MoE layer."""

    keep: list[list[int]]
    """Surviving expert ids per layer, ascending, in the profiler's layer order."""
    n_experts: int
    top_k: int
    layer_indices: list[int] = field(default_factory=list)
    """Each layer's index within its stack, as the profiling run recorded it."""
    module_names: list[str] = field(default_factory=list)
    """The MoE block module paths, used to match a plan onto a loaded model."""
    budget: dict = field(default_factory=dict)
    """How the allocation was made: mode, weighting, totals, floors."""
    source: dict = field(default_factory=dict)
    """Which run, and which of its generations, produced the scores."""
    diagnostics: dict = field(default_factory=dict)
    """Per-layer retention, block impact, and the uniform budget it beat."""
    created: str = ""

    @classmethod
    def from_allocation(cls, alloc: Allocation, *, n_experts: int, top_k: int,
                        layer_indices=None, module_names=None,
                        source: dict | None = None,
                        constraints: dict | None = None,
                        extra_diagnostics: dict | None = None) -> Self:
        """
        Turn an `Allocation` into the plan that records it.

        Args:
            alloc: the spent budget, as `moe_pruner.budget.allocate` returns it
            n_experts, top_k: the router's shape, which the allocation does not
                carry and every consumer of the plan needs
            layer_indices, module_names: what the run called these layers;
                without them the plan is only positional
            source: which run and which of its generations produced the scores
            constraints: what the allocator was *asked* for, as against what it
                came out as, which `budget` records
            extra_diagnostics: anything the caller measured that the allocation
                does not know about, merged into `diagnostics`

        Returns:
            the plan, ready to write or to cut a checkpoint with
        """
        L = alloc.n_keep.shape[0]
        return cls(
            keep=[alloc.keep_ids(l).tolist() for l in range(L)],
            n_experts=n_experts, top_k=top_k,
            layer_indices=list(map(int, layer_indices or range(L))),
            module_names=list(module_names or []),
            budget=cls._budget(alloc, n_experts, constraints),
            source=dict(source or {}),
            diagnostics=cls._diagnostics(alloc, extra_diagnostics))

    @staticmethod
    def _budget(alloc: Allocation, n_experts: int,
                constraints: dict | None) -> dict:
        """How the budget was sized and where it landed."""
        L, kept = alloc.n_keep.shape[0], alloc.total_kept
        return {"mode": alloc.mode, "weighting": alloc.weighting,
                "total_kept": kept, "n_layers": L,
                "fraction_kept": kept / (L * n_experts),
                # What the allocation came out as, not what it was told: the
                # constraints it was given are recorded separately.
                "min_kept": int(alloc.n_keep.min()),
                "max_kept": int(alloc.n_keep.max()),
                "constraints": dict(constraints or {})}

    @classmethod
    def _diagnostics(cls, alloc: Allocation, extra: dict | None) -> dict:
        """What the allocation is worth, per layer and against the uniform."""
        return {
            "weighted_retention": alloc.objective,
            "uniform_equivalent_experts": cls._uniform_equivalent(alloc),
            "per_layer_retention": [round(float(x), 6)
                                    for x in alloc.retained_per_layer],
            "per_layer_weight": [round(float(x), 8) for x in alloc.weight],
            "per_layer_n_keep": alloc.n_keep.tolist(),
            **(extra or {}),
        }

    @staticmethod
    def _uniform_equivalent(alloc: Allocation) -> int | None:
        """
        The uniform budget that reaches the same weighted retention, for scale.

        It is the number the greedy has to beat to have been worth running.
        None if no uniform count reaches it, the whole stack included.
        """
        L = alloc.n_keep.shape[0]
        uni = uniform_curve(alloc.retention, alloc.weight)
        reach = np.flatnonzero(uni >= alloc.objective)
        return int((reach[0] + 1) * L) if len(reach) else None

    @property
    def n_layers(self) -> int:
        return len(self.keep)

    @property
    def n_keep(self) -> np.ndarray:
        return np.array([len(k) for k in self.keep], dtype=np.int64)

    def mask(self) -> np.ndarray:
        """``[L, E]`` boolean: True where the expert survives."""
        m = np.zeros((self.n_layers, self.n_experts), dtype=bool)
        for l, ids in enumerate(self.keep):
            m[l, ids] = True
        return m

    def remap(self, layer: int) -> dict[int, int]:
        """Original expert id -> its index in the pruned layer."""
        return {int(e): i for i, e in enumerate(self.keep[layer])}

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "format_version": FORMAT_VERSION,
            "created": self.created or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "n_layers": self.n_layers,
            "total_kept": int(self.n_keep.sum()),
            "budget": self.budget,
            "source": self.source,
            "diagnostics": self.diagnostics,
            "layer_indices": self.layer_indices,
            "module_names": self.module_names,
            "keep": [list(map(int, ids)) for ids in self.keep],
        }, indent=2))
        return path

    @classmethod
    def read(cls, path: Path) -> Self:
        d = json.loads(Path(path).read_text())
        v = d.get("format_version")
        if v != FORMAT_VERSION:
            raise ValueError(f"{path} is a v{v} plan; this code reads "
                             f"v{FORMAT_VERSION}")
        return cls(keep=[list(map(int, ids)) for ids in d["keep"]],
                   n_experts=int(d["n_experts"]), top_k=int(d["top_k"]),
                   layer_indices=d.get("layer_indices", []),
                   module_names=d.get("module_names", []),
                   budget=d.get("budget", {}), source=d.get("source", {}),
                   diagnostics=d.get("diagnostics", {}),
                   created=d.get("created", ""))

    def write_mask_json(self, path: Path) -> Path:
        """
        Dump the plan as EASY-EP's ``[L][E]`` 0/1 matrix.

        Their tooling -- the sglang gate patch, ``model_prune.py`` -- eats this
        and nothing else, so it is the cheapest bridge to an existing pipeline.
        It cannot express which run or which budget produced it; the JSON plan
        is the record, this is an export.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.mask().astype(float).tolist()))
        return path
