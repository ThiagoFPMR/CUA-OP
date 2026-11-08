"""On-disk format of one generation's routing stats (a gen_*.npz), for L MoE layers,
T tokens and top-k routing:

  top_idx    [L, T, M]  expert ids, highest router logit first (M >= k)
  top_logit  [L, T, M]  their router logits; gate weights are recomputed on read
  norms      [L, T, k]  L2 norm of each selected expert's output
  impact     [L, T]     1 - cos(h, h + routed output), kept as 1 - cos for precision
"""
import json
from pathlib import Path

import numpy as np

#: Bumped whenever the array layout changes in a way readers must notice.
FORMAT_VERSION = 2

#: Expert-distribution columns kept per token when the caller does not say.
#: k is enough for pruning routed experts -- that is the whole selection, and
#: no analysis here reads past it. Raise it only to buy counterfactuals the
#: selection cannot answer, e.g. where a pruned expert's tokens would fall
#: back to; each extra column costs about 6% of the run.
DEFAULT_TOP_M = 8


def _idx_dtype(n_experts: int):
    """The narrowest unsigned type that can name every expert."""
    return np.uint8 if n_experts <= 256 else np.uint16


def pack_generation(
    scores: np.ndarray,
    norms: np.ndarray,
    sim: np.ndarray,
    n_experts: int,
    top_m: int = DEFAULT_TOP_M,
    scattered: bool = False,
) -> dict[str, np.ndarray]:
    """
    Reduce one generation's raw probe output to the stored arrays.

    Args:
        scores: ``[L, T, E]`` router logits, full precision.
        norms: ``[L, T, k]`` L2 norm of each selected expert's output.
        sim: ``[L, T]`` cos(h, h + routed_delta).
        n_experts: E, which decides the index dtype.
        top_m: How many experts to keep per token. Clamped to E.
        scattered: True when the router hid its pre-top-k distribution and
            `scores` is a scatter of the top-k *weights* rather than logits.
            Then the stored values are already the weights, and the reader
            must not push them through a softmax a second time.

    Returns:
        The arrays to hand to ``np.savez_compressed``.
    """
    scores = np.asarray(scores, dtype=np.float32)
    m = min(top_m, scores.shape[-1])

    # argpartition puts the m largest in front, unordered; sort just those.
    part = np.argpartition(-scores, m - 1, axis=-1)[..., :m]
    vals = np.take_along_axis(scores, part, axis=-1)
    order = np.argsort(-vals, axis=-1)

    return pack_selected(np.take_along_axis(part, order, -1),
                         np.take_along_axis(vals, order, -1),
                         norms, sim, n_experts=n_experts, scattered=scattered)


def pack_selected(
    top_idx: np.ndarray,
    top_logit: np.ndarray,
    norms: np.ndarray,
    sim: np.ndarray,
    n_experts: int,
    scattered: bool = False,
) -> dict[str, np.ndarray]:
    """
    `pack_generation` for stats whose top-m columns are already selected.

    Args:
        top_idx: ``[L, T, M]`` expert ids, descending by router logit.
        top_logit: ``[L, T, M]`` the router logits at those experts.
        norms, sim, n_experts, scattered: As for `pack_generation`.

    Returns:
        The arrays to hand to ``np.savez_compressed``.
    """
    return {
        "top_idx": np.asarray(top_idx).astype(_idx_dtype(n_experts)),
        "top_logit": np.asarray(top_logit, dtype=np.float32).astype(np.float16),
        "norms": np.asarray(norms, dtype=np.float32).astype(np.float16),
        # Stored as the complement; see the module docstring.
        "impact": (1.0 - np.asarray(sim, dtype=np.float64)).astype(np.float16),
        # A scatter has zeros everywhere outside the top-k, so only the first k
        # columns mean anything; the flag tells the reader that, and that the
        # values are weights already.
        "scattered": np.array(bool(scattered)),
        "n_experts": np.array(int(n_experts)),
    }


def write_generation(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write one generation's arrays to ``path`` (a ``.npz``)."""
    np.savez_compressed(path, format_version=np.array(FORMAT_VERSION), **arrays)


def read_generation(path: Path, top_k: int) -> dict[str, np.ndarray]:
    """
    Read one generation back, re-deriving what the format does not store.

    Args:
        path: The ``gen_*.npz`` to read.
        top_k: Experts per token, k. The stored arrays keep M >= k columns;
            the selection is the first k of them.

    Returns:
        ``{"idxs", "weights", "norms", "sim", "impact", "top_idx",
        "top_logit"}``. ``impact`` is ``1 - sim`` served directly, which is
        what analysis actually wants and what the file actually stores.
    """
    with np.load(path) as z:
        top_idx, top_logit = z["top_idx"], z["top_logit"]
        norms, impact = z["norms"], z["impact"]
        scattered = bool(z["scattered"])

    idxs = top_idx[..., :top_k].astype(np.int16)
    sel = top_logit[..., :top_k].astype(np.float32)
    if scattered:
        # The router never exposed logits; these are the weights themselves.
        weights = sel
    else:
        # weights = softmax over the selected logits. Subtracting the row max
        # is the usual overflow guard and cancels out of the ratio.
        e = np.exp(sel - sel.max(axis=-1, keepdims=True))
        weights = e / e.sum(axis=-1, keepdims=True)

    return {
        "idxs": idxs,
        "weights": weights,
        "norms": norms[..., :top_k].astype(np.float32),
        "sim": 1.0 - impact.astype(np.float32),
        "impact": impact.astype(np.float32),
        "top_idx": top_idx,
        "top_logit": top_logit,
    }


def read_outputs(path: Path) -> list[dict]:
    """
    Read an ``*.outputs.jsonl``, undoing the incremental prompt encoding.

    Successive prompts in a trajectory share a long prefix -- the next step
    replays the same system header and history -- so each row stores only the
    tokens past its predecessor's prefix.
    """
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    prev: list[int] = []
    for row in rows:
        shared = row.pop("prompt_shared_prefix")
        row["prompt_token_ids"] = prev[:shared] + row.pop("prompt_token_ids_suffix")
        prev = row["prompt_token_ids"]
    return rows
