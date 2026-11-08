"""Find the sparse-MoE blocks structurally (no hardcoded module paths)."""
import re
import torch.nn as nn

ROUTER_ATTRS = ("gate", "router")
LAYER_RE = re.compile(r"^(?P<stack>.*)\blayers\.(?P<layer>\d+)\b")


def router_of(module):
    """Return (router_module, attr_name) if the block owns a router, else (None, None)."""
    for attr in ROUTER_ATTRS:
        r = getattr(module, attr, None)
        if isinstance(r, nn.Module):
            return r, attr
    return None, None


def find_moe_blocks(
    model, 
    scope: str | None = None
) -> list[tuple[str, str, int, nn.Module]]:
    """
    Find the sparse-MoE blocks in the model.

    A block is any submodule owning BOTH a router submodule (gate or
    router) and an experts attribute. 
    
    Blocks are grouped by "stack": the part of the module path before the
    first "layers.<L>", e.g.

        model.language_model.layers.3.mlp
            -> stack "model.language_model", layer 3
        model.vision_tower.encoder.layers.3.mlp
            -> stack "model.vision_tower.encoder", layer 3

    Each stack numbers its layers from 0, so grouping this way keeps a VLM
    whose vision tower also carries MoE layers from interleaving its two
    runs of blocks under one shared set of layer numbers.

    Args:
        model: The model to inspect.
        scope: Optional regex to filter which stacks to return.
    
    Returns:
        A list of (stack, module_name, layer_index, module) tuples, sorted by
        stack and layer_index. The layer_index is the index of the block within
        its stack, starting from 0.
    """
    found: list[tuple[str, str, int, nn.Module]] = []  # (stack, name, layer, mod)
    for name, module in model.named_modules():
        router, _ = router_of(module)
        experts = getattr(module, "experts", None)
        if router is None or experts is None:
            continue
        if not isinstance(experts, nn.Module):
            continue
        m = LAYER_RE.match(name)
        stack = m.group("stack") if m else name.rsplit(".", 1)[0]
        layer = int(m.group("layer")) if m else len(found)
        found.append((stack, name, layer, module))

    if not found:
        raise RuntimeError(
            "No MoE blocks found. Inspect model.named_modules(): expected a "
            "module exposing a `gate`/`router` submodule and an `experts` "
            "attribute.")

    stacks: dict[str, list] = {}
    for stack, name, layer, module in found:
        stacks.setdefault(stack, []).append((name, layer, module))

    if scope:
        pat = re.compile(scope)
        picked = {s: b for s, b in stacks.items() if pat.search(s)}
        if not picked:
            raise RuntimeError(f"--moe-scope {scope!r} matched none of: "
                               f"{sorted(stacks)}")
        stacks = picked
    elif len(stacks) > 1:
        # More than one stack has MoE layers (e.g. a MoE vision tower next to a
        # MoE text stack). Pruning targets the big one; say so out loud.
        best = max(stacks, key=lambda s: len(stacks[s]))
        print(f"[hooks] {len(stacks)} MoE stacks found "
              f"{ {s: len(b) for s, b in stacks.items()} }; using {best!r}. "
              f"Override with --moe-scope.", flush=True)
        stacks = {best: stacks[best]}

    blocks = [b for stack in sorted(stacks) for b in sorted(stacks[stack], key=lambda t: t[1])]
    layers = [l for _, l, _ in blocks]
    print(f"[hooks] {len(blocks)} MoE blocks, layers {layers[0]}..{layers[-1]}"
          f"{' (non-contiguous)' if layers != list(range(layers[0], layers[-1] + 1)) else ''}",
          flush=True)
    
    return blocks
