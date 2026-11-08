"""Apply a pruning plan to a loaded model without changing its weights.

Router hooks set the dropped experts' logits to -inf, so `pruned(model, plan)` gives
exactly the outputs of the pruned checkpoint `surgery` would write.
"""
from contextlib import contextmanager

import torch

from moe_profiler.discovery import find_moe_blocks, router_of
from moe_profiler.routing import INT_DTYPES

from .plan import PruningPlan

#: How far a candidate recipe's weights may sit from the router's own before
#: it is rejected. Generous: the router computes in float and casts to bf16.
_RECIPE_TOL = 1e-2


def _match(blocks, plan: PruningPlan):
    """Line the plan's layers up with the model's blocks, by name then order."""
    names = [n for n, _, _ in blocks]
    if plan.module_names:
        by_name = {n: i for i, n in enumerate(names)}
        if all(n in by_name for n in plan.module_names):
            return [by_name[n] for n in plan.module_names]
        print(f"[mask] the plan names modules this model does not have "
              f"(e.g. {next(n for n in plan.module_names if n not in by_name)!r}); "
              f"falling back to layer order", flush=True)
    if len(blocks) != plan.n_layers:
        raise ValueError(f"the plan covers {plan.n_layers} MoE layers but this "
                         f"model exposes {len(blocks)}; pass `scope`, or "
                         f"check that this is the checkpoint that was profiled")
    return list(range(len(blocks)))


def _select(logits: torch.Tensor, top_k: int, scoring: str, norm: bool):
    """One candidate reading of what a router does to its logits."""
    scores = logits.float().sigmoid() if scoring == "sigmoid" else logits.float().softmax(-1)
    w, i = torch.topk(scores, top_k, dim=-1)
    if norm:
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-20)
    return w.to(logits.dtype), i


def _slots(output, n_experts: int):
    """Which elements of a router's tuple are the logits, weights and indices."""
    logits_i = idxs_i = weights_i = None
    for j, t in enumerate(output):
        if not torch.is_tensor(t):
            continue
        if t.dtype in INT_DTYPES and t.ndim == 2:
            idxs_i = j
        elif t.ndim == 2 and t.shape[-1] == n_experts and logits_i is None:
            logits_i = j
    if idxs_i is not None:
        for j, t in enumerate(output):
            if (torch.is_tensor(t) and j not in (idxs_i, logits_i)
                    and t.shape == output[idxs_i].shape):
                weights_i = j
                break
    return logits_i, weights_i, idxs_i


class _RouterMask:
    """Forward hook making the dropped experts unselectable, whatever is returned."""

    def __init__(self, keep: torch.Tensor, name: str):
        self.keep = keep
        self.name = name
        self.recipe: tuple[str, bool] | None = None

    def _calibrate(self, logits, weights, idxs) -> tuple[str, bool]:
        """Find the reading of this router that reproduces its own output."""
        best, tried = None, []
        for scoring in ("softmax", "sigmoid"):
            for norm in (True, False):
                w, i = _select(logits, idxs.shape[-1], scoring, norm)
                if not torch.equal(i, idxs):
                    continue
                err = (w.float() - weights.float()).abs().max().item()
                tried.append(f"{scoring}{'+norm' if norm else ''}: {err:.2e}")
                if best is None or err < best[0]:
                    best = (err, scoring, norm)
        if best is None or best[0] > _RECIPE_TOL:
            raise RuntimeError(
                f"cannot reproduce {self.name}'s own routing from its logits "
                f"(tried {tried or 'nothing -- no candidate matched its indices'}); "
                f"masking would change the selection rule, so this router needs "
                f"its own adapter")
        return best[1], best[2]

    def __call__(self, module, args, output):
        keep = self.keep.to(output.device if torch.is_tensor(output) else
                            next(t.device for t in output if torch.is_tensor(t)))
        E = keep.shape[0]

        # The simple case: the router hands the block bare logits.
        if torch.is_tensor(output):
            if output.shape[-1] != E:
                raise RuntimeError(f"{self.name} returned {output.shape[-1]} scores "
                                   f"but the plan is for {E} experts")
            return output.masked_fill(~keep, float("-inf"))

        if not isinstance(output, (tuple, list)):
            raise RuntimeError(
                f"{self.name} returns {type(output).__name__}, which this hook "
                f"cannot rewrite; masking has to happen before the top-k")

        logits_i, weights_i, idxs_i = _slots(output, E)
        if logits_i is None:
            raise RuntimeError(
                f"{self.name} returns no [T, {E}] logits, only its own selection; "
                f"there is nothing to mask upstream of the top-k")
        logits = output[logits_i]
        if idxs_i is None:                      # logits plus extras: mask and go
            new = list(output)
            new[logits_i] = logits.masked_fill(~keep, float("-inf"))
            return type(output)(new)

        if self.recipe is None:
            self.recipe = self._calibrate(logits, output[weights_i], output[idxs_i])
        masked = logits.masked_fill(~keep, float("-inf"))
        w, i = _select(masked, output[idxs_i].shape[-1], *self.recipe)

        new = list(output)
        new[logits_i] = masked
        new[idxs_i] = i.to(output[idxs_i].dtype)
        if weights_i is not None:
            new[weights_i] = w.to(output[weights_i].dtype)
        return type(output)(new)


def attach(model, plan: PruningPlan, scope: str | None = None) -> list:
    """
    Mute every pruned expert on a loaded model.

    Args:
        model: The loaded MoE model.
        plan: The pruning plan to enforce.
        scope: Regex picking which module subtree's MoE blocks to hook; the
            same flag the profiler takes, and it must pick the same stack.

    Returns:
        The hook handles, to ``.remove()`` when done. Prefer `pruned`.
    """
    blocks = find_moe_blocks(model, scope)
    picks = _match(blocks, plan)
    mask = torch.from_numpy(plan.mask())

    handles = []
    for plan_layer, block_i in enumerate(picks):
        name, _, module = blocks[block_i]
        router, _attr = router_of(module)
        if router is None:
            raise RuntimeError(f"{name} has no router to mask")
        handles.append(router.register_forward_hook(
            _RouterMask(mask[plan_layer], name)))

    kept = int(plan.n_keep.sum())
    total = plan.n_layers * plan.n_experts
    print(f"[mask] {len(handles)} routers masked; {kept:,}/{total:,} experts "
          f"live ({100 * kept / total:.1f}%), {plan.n_keep.min()}-"
          f"{plan.n_keep.max()} per layer", flush=True)
    if plan.n_keep.min() < plan.top_k:
        print(f"[mask] WARNING: a layer keeps {plan.n_keep.min()} experts but the "
              f"router selects {plan.top_k}; it will re-select the same expert or "
              f"read past the survivors", flush=True)
    return handles


@contextmanager
def pruned(model, plan: PruningPlan, scope: str | None = None):
    """Run a block with the plan enforced, then put the model back."""
    handles = attach(model, plan, scope)
    try:
        yield model
    finally:
        for h in handles:
            h.remove()
