"""Decide how many experts each layer keeps.

`uniform` keeps the same number in every layer. `greedy` maximizes
sum_l w_l * R_l(n_l), where R_l(n) is the share of layer l's importance held by its
top n experts and w_l weighs the layer (`weighting`: impact, mass or flat).
"""
import heapq
from dataclasses import dataclass

import numpy as np

WEIGHTINGS = ("impact", "mass", "flat")


def layer_weights(scores: np.ndarray, block_impact: np.ndarray,
                  weighting: str) -> np.ndarray:
    """The per-layer ``w_l`` of the objective, normalized to sum to 1."""
    if weighting == "impact":
        w = np.asarray(block_impact, dtype=np.float64)
    elif weighting == "mass":
        w = scores.sum(1).astype(np.float64)
    elif weighting == "flat":
        w = np.ones(scores.shape[0], dtype=np.float64)
    else:
        raise ValueError(f"unknown weighting {weighting!r}; expected one of {WEIGHTINGS}")
    total = w.sum()
    if total <= 0:
        raise ValueError(f"the {weighting!r} layer weights are all zero")
    return w / total


@dataclass
class Allocation:
    """A budget spent, plus everything needed to explain or replot it."""

    n_keep: np.ndarray        # [L]      experts kept in each layer
    order: np.ndarray         # [L, E]   expert ids, descending by importance
    retention: np.ndarray     # [L, E]   R_l(n): within-layer cumulative share
    weight: np.ndarray        # [L]      the w_l used, summing to 1
    curve_total: np.ndarray   # [M]      experts kept, along the greedy sweep
    curve_score: np.ndarray   # [M]      J at each of those totals
    mode: str
    weighting: str

    @property
    def total_kept(self) -> int:
        return int(self.n_keep.sum())

    @property
    def objective(self) -> float:
        """J(n): the weighted share of importance the allocation retains."""
        return float(self.retained_per_layer @ self.weight)

    @property
    def retained_per_layer(self) -> np.ndarray:
        """``R_l(n_l)`` for the chosen budget: each layer's own retained share."""
        return self.retention[np.arange(len(self.n_keep)), self.n_keep - 1]

    def keep_ids(self, layer: int) -> np.ndarray:
        """The surviving expert ids in a layer, in ascending original order."""
        return np.sort(self.order[layer, :self.n_keep[layer]])

    def mask(self) -> np.ndarray:
        """``[L, E]`` boolean: True where the expert survives."""
        m = np.zeros(self.order.shape, dtype=bool)
        for l, n in enumerate(self.n_keep):
            m[l, self.order[l, :n]] = True
        return m


def _rank(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort each layer's experts by importance and build its retention curve.

    Args:
        scores: ``[L, E]`` expert importance, non-negative.

    Returns:
        order: ``[L, E]`` expert ids, descending by importance.
        retention: ``[L, E]`` cumulative share of importance, per layer.
    """
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked_scores = np.take_along_axis(scores.astype(np.float64), order, axis=1)
    total = ranked_scores.sum(1, keepdims=True)
    # A layer nothing was ever routed to has no curve; leave it flat at zero so
    # it attracts no budget beyond its floor rather than dividing by nothing.
    retention = np.divide(ranked_scores, total, out=np.zeros_like(ranked_scores), where=total > 0).cumsum(1)
    return order, retention


def _sweep(retention: np.ndarray, weight: np.ndarray,
           n_min: np.ndarray, n_max: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the greedy curve to exhaustion, recording the whole trajectory.
    M is the number of experts the greedy can hand out beyond the floors.

    Args:
        retention: ``[L, E]`` cumulative share of importance, per layer.
        weight: ``[L]`` the w_l used, summing to 1.
        n_min: ``[L]`` floor per layer.
        n_max: ``[L]`` ceiling per layer.

    Returns:
        picks: ``[M]`` the layer that got each next expert, in order.
        curve: ``[M]`` the weighted retention after each pick.
    """
    L, E = retention.shape
    # Gain is how much J(n) rises if the greedy gives the next expert to each layer.
    # Computed as w_l . (R_l(n+1) - R_l(n)) for each layer and each n
    gain = np.empty_like(retention)
    gain[:, 0] = retention[:, 0]
    gain[:, 1:] = retention[:, 1:] - retention[:, :-1]
    gain *= weight[:, None]

    n = n_min.copy()
    running_objective = float(weight @ retention[np.arange(L), n - 1]) # J(n_min)
    picks: list[int] = []
    curve = [running_objective]
    # Initialize heap with the first gain for each layer that can still take more experts.
    heap = [(-gain[l, n[l]], l) for l in range(L) if n[l] < n_max[l]]
    heapq.heapify(heap)
    while heap:
        g, layer = heapq.heappop(heap)
        n[layer] += 1
        running_objective -= g
        picks.append(layer)
        curve.append(running_objective)
        # If the layer can still take more experts, push its next gain onto the heap.
        if n[layer] < n_max[layer]:
            heapq.heappush(heap, (-gain[layer, n[layer]], layer))
    return np.asarray(picks, dtype=np.int64), np.asarray(curve)


def uniform_curve(retention: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    The weighted retention for the uniform budget, as a function of the per-layer count N.
    
    It calculates J(N) = sum_l w_l . R_l(N) for N = 1..E, where R_l(N) is the cumulative 
    share of importance retained in layer l when keeping the top N experts.
    
    Args:
        retention: ``[L, E]`` cumulative share of importance, per layer.
        weight: ``[L]`` the w_l used, summing to 1.
    
    Returns:
        ``[E]`` the weighted retention if every layer keeps the same count N.
    """
    return retention.T @ weight            # [E], indexed by N - 1


#: Retention levels the trade-off table reports, for reading a budget off it.
RETENTION_LEVELS = (0.80, 0.90, 0.95, 0.99)


def tradeoff(alloc: "Allocation", levels=RETENTION_LEVELS) -> list[dict]:
    """
    The budget each retention level costs, greedily and uniformly.

    The greedy sweep already walks every budget from the floor to the whole
    stack, recording J at each step, so this is a read-off rather than a
    computation.

    Args:
        alloc: The allocation to report on.
        levels: The retention levels to report on.
    
    Returns:
        A list of dicts, one per level, with keys:
        retention, greedy, uniform, total.
    """
    L, E = alloc.retention.shape
    uni = uniform_curve(alloc.retention, alloc.weight)
    rows = []
    for t in levels:
        g = np.flatnonzero(alloc.curve_score >= t)
        u = np.flatnonzero(uni >= t)
        rows.append({
            "retention": float(t),
            "greedy": int(alloc.curve_total[g[0]]) if len(g) else None,
            "uniform": int((u[0] + 1) * L) if len(u) else None,
            "total": L * E,
        })
    return rows



def curve_points(alloc: "Allocation", n: int = 50) -> list[list[float]]:
    """``[[experts kept, retention], ...]``, downsampled enough to store."""
    idx = np.unique(np.linspace(0, len(alloc.curve_score) - 1, n).astype(int))
    return [[int(alloc.curve_total[i]), round(float(alloc.curve_score[i]), 6)]
            for i in idx]


def allocate(scores: np.ndarray, layer_impact: np.ndarray, *,
             mode: str = "greedy", weighting: str = "impact",
             total: int | None = None, per_layer: int | None = None,
             fraction: float | None = None, target: float | None = None,
             min_experts: int = 8, max_experts: int | None = None) -> Allocation:
    """
    Choose how many experts each layer keeps, and which.

    Exactly one of ``total`` / ``per_layer`` / ``fraction`` / ``target`` says
    how big the budget is; ``target`` names the weighted retention to reach and
    lets the allocator find the smallest budget that reaches it.

    Args:
        scores: ``[L, E]`` expert importance, non-negative.
        layer_impact: ``[L]`` mean per-token ``s_t``, for the ``impact`` weighting.
        mode: ``greedy`` for the impact-weighted allocation, ``uniform`` for
            EASY-EP's equal count in every layer.
        weighting: Which ``w_l`` the greedy optimises; see the module docstring.
            Ignored by ``uniform``, which has no cross-layer decision to make.
        min_experts: Floor per layer. Must be at least top-k or the router
            cannot fill its selection.
        max_experts: Ceiling per layer; default E.

    Returns:
        An `Allocation`, whose ``keep_ids`` and ``mask`` are the pruning decision.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"scores must be [L, E], got {scores.shape}")
    if (scores < 0).any():
        raise ValueError("expert importance must be non-negative")
    L, E = scores.shape

    given = [n for n in (total, per_layer, fraction, target) if n is not None]
    if len(given) != 1:
        raise ValueError("give exactly one of total / per_layer / fraction / target")

    hi = int(max_experts or E)
    lo = int(min_experts)
    if not 1 <= lo <= hi <= E:
        raise ValueError(f"need 1 <= min_experts <= max_experts <= {E}, "
                         f"got {lo} and {hi}")
    n_min = np.full(L, lo, dtype=np.int64)
    n_max = np.full(L, hi, dtype=np.int64)

    order, retention = _rank(scores)
    w = layer_weights(scores, layer_impact, weighting)

    if mode == "uniform":
        curve = uniform_curve(retention, w)
        counts = np.arange(1, E + 1)
        if per_layer is not None:
            n = int(per_layer)
        elif total is not None:
            n = int(round(total / L))
        elif fraction is not None:
            n = int(round(fraction * E))
        else:
            reach = np.flatnonzero(curve >= target)
            if not len(reach):
                raise ValueError(f"even keeping every expert only reaches "
                                 f"{curve[-1]:.4f} weighted retention, below "
                                 f"target={target}")
            n = int(reach[0]) + 1
        n = int(np.clip(n, lo, hi))
        return Allocation(n_keep=np.full(L, n, dtype=np.int64), order=order,
                          retention=retention, weight=w,
                          curve_total=counts * L, curve_score=curve,
                          mode=mode, weighting=weighting)

    if mode != "greedy":
        raise ValueError(f"unknown mode {mode!r}; expected 'greedy' or 'uniform'")

    picks, curve = _sweep(retention, w, n_min, n_max)
    base = int(n_min.sum())
    curve_total = base + np.arange(len(curve))

    if target is not None:
        reach = np.flatnonzero(curve >= target)
        if not len(reach):
            raise ValueError(f"even keeping every expert only reaches {curve[-1]:.4f} "
                             f"weighted retention, below target={target}")
        steps = int(reach[0])
    else:
        if per_layer is not None:
            want = int(per_layer) * L
        elif fraction is not None:
            want = int(round(fraction * L * E))
        else:
            want = int(total)
        if want < base:
            raise ValueError(f"a budget of {want} cannot hold the floor of "
                             f"{lo} experts in each of {L} layers ({base}); "
                             f"lower min_experts or raise the budget")
        steps = min(want, base + len(picks)) - base

    n = n_min + np.bincount(picks[:steps], minlength=L)
    return Allocation(n_keep=n.astype(np.int64), order=order, retention=retention,
                      weight=w, curve_total=curve_total, curve_score=curve,
                      mode=mode, weighting=weighting)
