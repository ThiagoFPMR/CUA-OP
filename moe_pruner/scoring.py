"""Score each expert's importance from profiling runs.

For expert i of a MoE layer, over the tokens t routed to it:

    S_i = 1/N_i^b * sum_t  g_{i,t}^alpha * ||e_{i,t}||^beta * s_t^gamma

N_i is the number of tokens routed to i, g_{i,t} the gate weight t gave i
(renormalized over the top-k), ||e_{i,t}|| the norm of i's output for t, and
s_t = max(1 - cos(h_t, h_t + routed output), 0) how far the layer's experts moved t.
Named criteria are points (b, alpha, beta, gamma) of this family, listed in
`PRESETS`; the default is REAP, (1, 1, 1, 0).
"""
import dataclasses
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from moe_profiler.run import Run

#: Bumped when the cached array layout changes in a way readers must notice.
CACHE_VERSION = 2

#: Model-level run metadata a set of scores carries, so a cache read back
#: months later still says what model it came from. Every run merged into one
#: set of scores must agree on these.
META_KEYS = ("model_path", "model_type", "n_experts", "top_k",
             "layer_indices", "module_names")

#: Keys whose disagreement means two sets of scores are not of the same stack.
_STACK_KEYS = ("n_experts", "top_k", "layer_indices", "module_names")

#: Per-run metadata recorded in each set of scores' ``meta["runs"]``: where
#: the generations came from. These differ between the runs of one prune.
RUN_KEYS = ("episode_id", "task_id", "domain", "score", "stop_reason")


@dataclass(frozen=True)
class Criterion:
    """
    One point ``(b, alpha, beta, gamma)`` of the score family above.

    ``alpha``, ``beta`` and ``gamma`` act per token, so they are fixed when a
    run is scored. ``b`` only divides the finished sums by the routed-token
    counts, so it can change on a cached set of scores for free.
    """
    # Defaults are REAP's, the default criterion.
    b: float = 1.0       # routing frequency: divide by N_i^b
    alpha: float = 1.0   # gate weight exponent
    beta: float = 1.0    # activation (expert-output norm) exponent
    gamma: float = 0.0   # token contribution (1 - cos) exponent

    def __post_init__(self):
        bad = {k: v for k, v in dataclasses.asdict(self).items() if v < 0}
        if bad:
            # a negative exponent on a quantity that can be 0 is a division by it
            raise ValueError(f"score exponents must be non-negative, got {bad}")

    @property
    def token_exponents(self) -> tuple[float, float, float]:
        """What a run must be scored with: everything but ``b``."""
        return (self.alpha, self.beta, self.gamma)

    @property
    def name(self) -> str | None:
        """The preset this criterion is, if it is one."""
        return next((k for k, v in PRESETS.items() if v == self), None)

    @property
    def degenerate(self) -> bool:
        """Every routed expert scores 1: a ranking by nothing (the paper's N/A)."""
        return self.b == 1 and self.token_exponents == (0, 0, 0)

    def __str__(self) -> str:
        s = (f"S(b={self.b:g}, alpha={self.alpha:g}, "
             f"beta={self.beta:g}, gamma={self.gamma:g})")
        return f"{s} [{self.name}]" if self.name else s

    def as_dict(self) -> dict:
        return dataclasses.asdict(self) | {"name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> "Criterion":
        return cls(**{f.name: float(d[f.name]) for f in dataclasses.fields(cls)
                      if f.name in d})


#: The published criteria, as points of the family. REAP is the default.
PRESETS: dict[str, Criterion] = {
    "easy-ep":   Criterion(b=0, alpha=1, beta=1, gamma=1),
    "frequency": Criterion(b=0, alpha=0, beta=0, gamma=0),
    "seer":      Criterion(b=0, alpha=1, beta=0, gamma=0),
    "ean":       Criterion(b=0, alpha=0, beta=1, gamma=0),
    "reap":      Criterion(b=1, alpha=1, beta=1, gamma=0),
    "man":       Criterion(b=1, alpha=0, beta=1, gamma=0),
    "msan":      Criterion(b=1, alpha=0, beta=2, gamma=0),
}

DEFAULT_CRITERION = PRESETS["reap"]


def _pow(x: np.ndarray, p: float) -> np.ndarray:
    """``x ** p``, without the work when the exponent is 1."""
    return x if p == 1 else x ** p


@dataclass
class ExpertScores:
    """
    Per-layer, per-expert importance accumulated over a run.

    Held as accumulators rather than a finished ranking so one expensive pass
    answers every later question that does not reach inside a token: the
    frequency normalization ``b``, and how to weight one layer against another,
    are decisions taken on these arrays and not on the run.

    The accumulation itself is `update`, one generation at a time, and every
    field is a plain sum -- which is what lets `merge` combine two passes over
    different slices of a run, or over different runs of the same model.
    """
    criterion: Criterion
    score: np.ndarray        # [L, E]  sum_t g^a . ||e||^B . s^y
    count: np.ndarray        # [L, E]  N_i: times the expert was routed to
    impact_sum: np.ndarray   # [L]     sum_t s_t
    impact_cnt: int          #         tokens behind that sum, all layers alike
    n_generations: int
    meta: dict

    @classmethod
    def zeros(cls, n_layers: int, n_experts: int,
              criterion: Criterion = DEFAULT_CRITERION,
              meta: dict | None = None) -> "ExpertScores":
        """An empty accumulator of a known shape, ready for `update`."""
        return cls(criterion=criterion,
                   score=np.zeros((n_layers, n_experts)),
                   count=np.zeros((n_layers, n_experts), dtype=np.int64),
                   impact_sum=np.zeros(n_layers),
                   impact_cnt=0, n_generations=0, meta=meta or {})

    @property
    def n_layers(self) -> int:
        return self.score.shape[0]

    @property
    def n_experts(self) -> int:
        return self.score.shape[1]

    def update(self, record: dict, gen_idx: int | None = None) -> None:
        """
        Fold one generation's routing records into scores.

        Args:
            record: One generation as `moe_profiler.run.Run.generations` yields it:
                `idxs`/`weights`/`norms` as [L, T, k], `impact` as [L, T].
            gen_idx: The generation's index, named in any error raised here.
        """
        idxs = record["idxs"].astype(np.int64)                  # [L, T, k]
        weights = record["weights"].astype(np.float64)          # [L, T, k]
        norms = record["norms"].astype(np.float64)              # [L, T, k]
        # EASY-EP eq. 9, clamped: a routed delta too small to rotate the
        # residual can land on the wrong side of 1.0 in fp16 and must not vote
        # negatively.
        s = np.clip(record["impact"].astype(np.float64), 0.0, None)  # [L, T]

        L, E = self.score.shape
        where = f"generation {gen_idx}" if gen_idx is not None else "this generation"
        if idxs.shape[0] != L:
            raise ValueError(f"{where} has {idxs.shape[0]} layers but these "
                             f"accumulators hold {L}; these are not one run")
        if idxs.max() >= E:
            # Out-of-range ids would wrap into the next layer's counters and
            # silently corrupt every number here, so refuse instead.
            raise ValueError(f"{where} routes to expert {int(idxs.max())} "
                             f"but the run declares {E} experts; its metadata is wrong")

        c = self.criterion
        term = _pow(weights, c.alpha) * _pow(norms, c.beta) * _pow(s, c.gamma)[:, :, None]
        # One bincount over the whole [L, T, k] at a flat (layer * E) + expert
        # address, so every layer's counters are filled in a single pass.
        flat = (np.arange(L, dtype=np.int64)[:, None, None] * E + idxs).ravel()
        size = L * E
        shape = (L, E)
        self.score += np.bincount(flat, term.ravel(), minlength=size).reshape(shape)
        self.count += np.bincount(flat, minlength=size).reshape(shape)

        self.impact_sum += s.sum(1)
        self.impact_cnt += s.shape[1]
        self.n_generations += 1

    def merge(self, other: "ExpertScores") -> "ExpertScores":
        """
        Add another set of accumulators into this one, in place.

        The two must cover the same stack -- scoring one run in shards, or two
        runs of the same model -- because a sum over layer l only means
        anything if it is the same layer l on both sides. And they must sum the
        same per-token quantity, or the total is of nothing.
        """
        if self.score.shape != other.score.shape:
            raise ValueError(f"cannot merge accumulators of shape "
                             f"{other.score.shape} into {self.score.shape}")
        if self.criterion.token_exponents != other.criterion.token_exponents:
            raise ValueError(f"cannot merge scores taken with {other.criterion} "
                             f"into scores taken with {self.criterion}")
        for key in _STACK_KEYS:
            mine, theirs = self.meta.get(key), other.meta.get(key)
            if mine is not None and theirs is not None and mine != theirs:
                raise ValueError(f"cannot merge scores whose {key} differ "
                                 f"({theirs!r} into {mine!r}); they are not of "
                                 f"the same stack")
            if mine is None and theirs is not None:
                self.meta[key] = theirs
        self.meta["runs"] = self.meta.get("runs", []) + other.meta.get("runs", [])
        self.score += other.score
        self.count += other.count
        self.impact_sum += other.impact_sum
        self.impact_cnt += other.impact_cnt
        self.n_generations += other.n_generations
        return self

    def __iadd__(self, other: "ExpertScores") -> Self:
        return self.merge(other)

    def with_criterion(self, criterion: Criterion) -> "ExpertScores":
        """
        The same accumulators read under another ``b``.

        Only ``b`` can change after the fact; the other three exponents are
        inside the per-token sum, so changing them means scoring the run again.
        """
        if criterion.token_exponents != self.criterion.token_exponents:
            raise ValueError(
                f"these scores were taken with {self.criterion}; {criterion} "
                f"differs inside the per-token sum (alpha, beta, gamma), so "
                f"score the run again rather than reusing them")
        return dataclasses.replace(self, criterion=criterion)

    def importance(self) -> np.ndarray:
        """
        ``S_i``: the per-layer, per-expert importance the budget allocator spends on.

        Returns:
            A 2D numpy array of shape [L, E]: the accumulated sum divided by
            ``N_i^b``. An expert never routed to scores 0 whatever ``b`` is.
        """
        b = self.criterion.b
        if b == 0:
            return self.score
        return np.divide(self.score, self.count.astype(np.float64) ** b,
                         out=np.zeros_like(self.score), where=self.count > 0)

    def layer_impact(self) -> np.ndarray:
        """
        Mean per-token ``s_t`` for each layer: how far the block's routed
        experts move an average token's residual.

        This is the quantity the budget allocator weights layers by. It spans
        two orders of magnitude across a stack, and it is *not* recoverable
        from the importance scores, whose per-layer totals confound it with how
        loud that layer's experts happen to be. It does not depend on the
        criterion.
        """
        return self.impact_sum / max(self.impact_cnt, 1)

    @classmethod
    def from_run(cls, run: Run, criterion: Criterion = DEFAULT_CRITERION,
                 generations=None, progress_every: int = 10) -> "ExpertScores":
        """
        One streaming pass over a run, accumulating every quantity a budget needs.

        Args:
            run: An opened run (see `moe_profiler.run.Run.open`).
            criterion: The point of the score family to accumulate.
            generations: Restrict to these generation indices; None means all.
            progress_every: Print a line every N generations; 0 to stay quiet.

        Returns:
            The `ExpertScores` for the whole selection.
        """
        scores = None
        for gen_idx, record in run.generations(generations):
            if scores is None:
                # the stack's depth comes from its first generation
                meta = {k: run.meta.get(k) for k in META_KEYS}
                meta["runs"] = [{"stem": str(run.stem),
                                 **{k: run.meta.get(k) for k in RUN_KEYS}}]
                scores = cls.zeros(record["idxs"].shape[0], run.n_experts,
                                   criterion=criterion, meta=meta)
            scores.update(record, gen_idx)
            if progress_every and scores.n_generations % progress_every == 1:
                print(f"[score] {scores.n_generations}/{run.n_generations}  "
                      f"T={record['idxs'].shape[1]}", flush=True)

        if scores is None:
            raise ValueError("the run yielded no generations to score")
        scores.meta["runs"][0]["generations"] = scores.n_generations
        return scores

    @classmethod
    def from_runs(cls, runs: Iterable[Run],
                  criterion: Criterion = DEFAULT_CRITERION) -> "ExpertScores":
        """
        Score several runs of the same model and merge them into one set.

        The profiling server leaves one run per episode, so a prune for a
        domain is taken over many runs. Choosing which (a domain, the
        successful episodes only, ...) is the caller's business: every run's
        ``meta`` -- the episode's ``domain``, ``task_id``, ``score`` and so on --
        is there to select on.

        Args:
            runs: Opened runs (see `moe_profiler.run.Run.open`), all of one
                model and one MoE stack.
            criterion: The point of the score family to accumulate.

        Returns:
            The merged `ExpertScores`; ``meta["runs"]`` lists where each run's
            generations came from.
        """
        total = None
        runs = list(runs)
        for i, run in enumerate(runs):
            scores = cls.from_run(run, criterion, progress_every=0)
            total = scores if total is None else total.merge(scores)
            print(f"[score] run {i + 1}/{len(runs)}  {run.stem.parent.name}  "
                  f"{scores.n_generations} generations", flush=True)
        if total is None:
            raise ValueError("no runs to score")
        return total

    def save(self, path: Path) -> None:
        """
        Cache the class' accumulator attributes, so re-allocating a budget costs nothing.
        """
        np.savez_compressed(
            path, cache_version=np.array(CACHE_VERSION),
            criterion=np.array(json.dumps(self.criterion.as_dict())),
            score=self.score, count=self.count,
            impact_sum=self.impact_sum, impact_cnt=self.impact_cnt,
            n_generations=np.array(self.n_generations),
            meta=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path: Path) -> "ExpertScores":
        """
        Load a cached `ExpertScores` from disk.
        """
        with np.load(path, allow_pickle=False) as z:
            if "score" not in z.files:
                raise ValueError(f"{path} is not an expert-score cache "
                                 f"(has {sorted(z.files)})")
            version = int(z["cache_version"]) if "cache_version" in z.files else 0
            if version != CACHE_VERSION:
                raise ValueError(f"{path} is a version-{version} score cache and "
                                 f"this reads version {CACHE_VERSION}; score the "
                                 f"run again")
            return cls(criterion=Criterion.from_dict(json.loads(str(z["criterion"]))),
                       score=z["score"], count=z["count"],
                       impact_sum=z["impact_sum"], impact_cnt=int(z["impact_cnt"]),
                       n_generations=int(z["n_generations"]),
                       meta=json.loads(str(z["meta"])))
