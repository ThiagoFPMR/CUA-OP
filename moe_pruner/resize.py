"""Load a pruned checkpoint, including one with a different expert count per layer.

`load_pruned_model` builds each MoE block at its plan width before loading weights;
`resize_experts` narrows an already loaded model.
"""
import copy
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from moe_profiler.discovery import find_moe_blocks, router_of

from .masking import _match
from .plan import PruningPlan
from .surgery import SIDECAR

#: Attributes that hold the routed-expert count on a block or its submodules.
_COUNT_ATTRS = ("num_experts", "n_routed_experts", "num_local_experts")


def _narrow(param: nn.Parameter, keep: torch.Tensor) -> nn.Parameter:
    """Take ``keep`` along the expert axis, or just reshape on a meta tensor."""
    if param.device.type == "meta":
        shape = (len(keep),) + tuple(param.shape[1:])
        return nn.Parameter(torch.empty(shape, device="meta", dtype=param.dtype),
                            requires_grad=param.requires_grad)
    return nn.Parameter(param.detach().index_select(0, keep.to(param.device)),
                        requires_grad=param.requires_grad)


def _resize_module(module: nn.Module, keep: torch.Tensor, n_experts: int) -> int:
    """Narrow every ``[E, ...]`` parameter owned directly by ``module``."""
    n = 0
    for name, param in list(module.named_parameters(recurse=False)):
        if param.ndim >= 1 and param.shape[0] == n_experts:
            setattr(module, name, _narrow(param, keep))
            n += 1
    for attr in _COUNT_ATTRS:
        if getattr(module, attr, None) == n_experts:
            setattr(module, attr, len(keep))
    return n


def resize_experts(model: nn.Module, plan: PruningPlan,
                   scope: str | None = None) -> nn.Module:
    """
    Narrow every MoE block in ``model`` to the experts its plan keeps.

    Safe on a meta-device skeleton (shapes only) and on real weights (the
    surviving rows are selected), so the same call serves both the load path
    and shrinking a model that is already resident.

    Args:
        model: The loaded or skeletal MoE model.
        plan: The plan to enforce.
        scope: Regex picking which module subtree's MoE blocks to resize.

    Returns:
        ``model``, resized in place.
    """
    blocks = find_moe_blocks(model, scope)
    picks = _match(blocks, plan)
    touched = 0

    for plan_layer, block_i in enumerate(picks):
        name, _, block = blocks[block_i]
        keep = torch.tensor(plan.keep[plan_layer], dtype=torch.long)
        router, _attr = router_of(block)
        if router is None:
            raise RuntimeError(f"{name} has no router to resize")

        touched += _resize_module(router, keep, plan.n_experts)
        experts = block.experts
        if isinstance(experts, (nn.ModuleList, nn.Sequential)):
            # Per-expert modules: keep the survivors, in the plan's order, so
            # the list index becomes the new expert id.
            block.experts = type(experts)([experts[i] for i in plan.keep[plan_layer]])
            touched += len(plan.keep[plan_layer])
        else:
            touched += _resize_module(experts, keep, plan.n_experts)
        # The count often lives on the block itself as well as its children.
        for attr in _COUNT_ATTRS:
            if getattr(block, attr, None) == plan.n_experts:
                setattr(block, attr, len(keep))

    kept = int(plan.n_keep.sum())
    total = plan.n_layers * plan.n_experts
    print(f"[resize] {len(picks)} blocks, {touched} tensors narrowed; "
          f"{kept:,}/{total:,} experts ({100 * kept / total:.1f}%), "
          f"{plan.n_keep.min()}-{plan.n_keep.max()} per layer", flush=True)
    return model


def _factories():
    """The auto classes to try: the VLM head first, then plain causal LM."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    return [AutoModelForImageTextToText, AutoModelForCausalLM]


def _try_factories(build, what: str):
    """Run ``build`` against each auto class until one accepts the config."""
    errors = []
    for factory in _factories():
        try:
            model = build(factory)
            print(f"[load] {what} via {factory.__name__}", flush=True)
            return model
        except Exception as exc:  # wrong head for this config -- try the next  # noqa: BLE001
            errors.append(f"{factory.__name__}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"could not {what} for this config:\n  " + "\n  ".join(errors))


@contextmanager
def per_layer_widths(block_cls, counts, n_experts: int):
    """
    Make ``block_cls`` build its i-th instance with ``counts[i]`` experts.

    A non-uniform checkpoint has to be built before it can be filled, and the
    thing that decides an MoE block's width is the ``num_experts`` it reads out
    of the config in ``__init__``. There is exactly one config and one class,
    so the width is handed over by construction order instead: MoE blocks are
    built in layer order, and each one is given a shallow copy of the config
    carrying its own count.

    This is what buys ordinary `from_pretrained` loading -- with its sharding,
    dtype handling, device maps and, importantly, its checkpoint-to-module key
    conversion -- for an architecture the config cannot describe.
    """
    original = block_cls.__init__
    widths = iter(list(counts))

    def __init__(self, config, *args, **kwargs):
        try:
            n = next(widths)
        except StopIteration:
            raise RuntimeError(
                f"{block_cls.__name__} was built more times than the plan has "
                f"layers; the plan does not match this architecture") from None
        cfg = copy.copy(config)
        for attr in _COUNT_ATTRS:
            if getattr(cfg, attr, None) == n_experts:
                setattr(cfg, attr, n)
        original(self, cfg, *args, **kwargs)

    block_cls.__init__ = __init__
    try:
        yield
    finally:
        block_cls.__init__ = original


def _block_class(cfg, attn: str, scope: str | None):
    """Which class implements this model's MoE block, from a meta skeleton."""
    from accelerate import init_empty_weights

    def build(factory):
        with init_empty_weights():
            return factory.from_config(cfg, trust_remote_code=True,
                                       attn_implementation=attn)

    skeleton = _try_factories(build, "build a skeleton")
    blocks = find_moe_blocks(skeleton, scope)
    return type(blocks[0][2]), len(blocks)


def load_pruned_model(path: Path, plan: PruningPlan | None = None,
                      dtype=torch.bfloat16, device_map: str | None = "auto",
                      attn: str = "sdpa", scope: str | None = None):
    """
    Load a checkpoint written by `surgery.prune_checkpoint`.

    A uniform checkpoint needs nothing special: its config declares the new
    count, so the ordinary loader builds the right shapes. A non-uniform one
    goes through the same loader with `per_layer_widths` in force, so each
    block is constructed at its own width and everything downstream --
    sharding, dtype, device placement, key conversion -- is unchanged.

    Args:
        path: The pruned checkpoint directory.
        plan: Default: the ``expert_pruning.json`` the surgery left there.
        dtype, device_map, attn: As for `moe_profiler.loading`.
        scope: Regex picking which MoE stack the plan applies to.

    Returns:
        ``(model, processor)``; the processor is None for a text-only model.
    """
    from transformers import AutoConfig, AutoProcessor

    path = Path(path)
    plan = plan or PruningPlan.read(path / SIDECAR)
    counts = plan.n_keep.tolist()
    load = lambda f: f.from_pretrained(  # noqa: E731
        path, dtype=dtype, device_map=device_map, trust_remote_code=True,
        attn_implementation=attn)

    if len(set(counts)) == 1:
        print(f"[load] uniform plan ({counts[0]} experts everywhere); the config "
              f"describes it, so this is an ordinary checkpoint")
        model = _try_factories(load, "load")
    else:
        print(f"[load] non-uniform plan ({min(counts)}-{max(counts)} experts); "
              f"building each block at its own width")
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        block_cls, n_blocks = _block_class(cfg, attn, scope)
        if n_blocks != plan.n_layers:
            raise ValueError(f"the plan covers {plan.n_layers} MoE layers but this "
                             f"config builds {n_blocks}")
        print(f"[load] MoE block class {block_cls.__name__}", flush=True)
        with per_layer_widths(block_cls, counts, plan.n_experts):
            model = _try_factories(load, "load")

    model.eval()
    try:
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    except Exception as exc:  # a text-only checkpoint has none  # noqa: BLE001
        print(f"[load] no processor ({type(exc).__name__}); returning None")
        processor = None
    return model, processor
