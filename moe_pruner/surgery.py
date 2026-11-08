"""Write a copy of a checkpoint with the dropped experts removed.

The result matches `masking.pruned` exactly. A uniform plan updates the config's
expert count; a non-uniform checkpoint loads with `resize.load_pruned_model`.
"""
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from .plan import PruningPlan

#: Written beside a pruned checkpoint; the loader reads it back.
SIDECAR = "expert_pruning.json"

#: A per-expert weight lives under ``<stack>.layers.<n>.<block>.experts.``.
#: Group by the part before ``.layers.<n>`` so a MoE vision tower next to a MoE
#: text stack does not interleave under one shared set of layer numbers.
_EXPERT_RE = re.compile(r"^(?P<stack>.*)\.layers\.(?P<layer>\d+)\.(?P<block>[^.]+)\.experts\.")

#: Config keys that name the routed-expert count, across model families.
_COUNT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts")


def _read_shapes(src: Path) -> tuple[dict[str, list[int]], dict[str, str]]:
    """Every tensor's shape and which file holds it, without reading any data."""
    index = src / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        single = src / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"no model.safetensors or model.safetensors.index.json "
                             f"in {src}; only safetensors checkpoints can be pruned")
        with safe_open(single, framework="pt") as f:
            weight_map = {k: single.name for k in f.keys()}

    shapes: dict[str, list[int]] = {}
    for fname in sorted(set(weight_map.values())):
        with safe_open(src / fname, framework="pt") as f:
            for k in f.keys():
                shapes[k] = list(f.get_slice(k).get_shape())
    return shapes, weight_map


def _discover_blocks(shapes: dict[str, list[int]], n_experts: int,
                     module_names: list[str] | None) -> list[str]:
    """
    Find the MoE block key prefixes, in layer order, from the key names alone.

    Returns one prefix per layer, e.g. ``model.language_model.layers.7.mlp``.
    """
    stacks: dict[str, dict[int, str]] = defaultdict(dict)
    for key in shapes:
        m = _EXPERT_RE.match(key)
        if m:
            stacks[m.group("stack")][int(m.group("layer"))] = (
                f"{m.group('stack')}.layers.{m.group('layer')}.{m.group('block')}")
    if not stacks:
        raise ValueError("no `...layers.<n>.<block>.experts.` tensors in this "
                         "checkpoint; nothing here looks like a routed MoE")

    if len(stacks) > 1:
        named = {s for s in stacks
                 for n in (module_names or []) if n.startswith(s + ".")}
        if len(named) == 1:
            stacks = {s: stacks[s] for s in named}
        else:
            best = max(stacks, key=lambda s: len(stacks[s]))
            print(f"[surgery] {len(stacks)} MoE stacks "
                  f"{ {s: len(b) for s, b in stacks.items()} }; using {best!r}")
            stacks = {best: stacks[best]}

    blocks = next(iter(stacks.values()))
    return [blocks[l] for l in sorted(blocks)]


def _classify(prefix: str, shapes: dict[str, list[int]], n_experts: int):
    """
    Split one block's tensors into what the surgery does to them.

    Returns ``(sliced, listed, untouched)``:
      sliced    key -> take the plan's experts along axis 0 (fused layouts,
                and the router's own [E, ...] weights and biases)
      listed    key -> (expert id, tail), for a per-expert ``nn.ModuleList``
                layout, where surviving experts have to be renumbered
      untouched everything else in the block: shared experts, layernorms, the
                shared-expert gate
    """
    sliced: dict[str, None] = {}
    listed: dict[str, tuple[int, str]] = {}
    untouched: list[str] = []
    exp_prefix = f"{prefix}.experts."
    per_expert = re.compile(rf"^{re.escape(exp_prefix)}(\d+)\.(.+)$")

    for key, shape in shapes.items():
        if not key.startswith(prefix + "."):
            continue
        m = per_expert.match(key)
        if m:
            listed[key] = (int(m.group(1)), m.group(2))
        # A shared expert is dense and always on; it is never routed, so its
        # first axis is an intermediate size that may collide with E by chance.
        elif "shared_expert" in key:
            untouched.append(key)
        elif shape and shape[0] == n_experts:
            sliced[key] = None
        else:
            untouched.append(key)
    return sliced, listed, untouched


def _update_config(src: Path, dst: Path, plan: PruningPlan, uniform: int | None):
    """Point ``num_experts`` at the new count, wherever the config nests it."""
    path = src / "config.json"
    if not path.exists():
        print("[surgery] no config.json to update")
        return
    cfg = json.loads(path.read_text())
    hits: list[str] = []

    def walk(node, trail):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in _COUNT_KEYS and v == plan.n_experts:
                    if uniform is not None:
                        node[k] = uniform
                    hits.append(".".join(trail + [k]))
                else:
                    walk(v, trail + [k])
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, trail + [str(i)])

    walk(cfg, [])
    if not hits:
        print(f"[surgery] WARNING: no config key named {_COUNT_KEYS} holds "
              f"{plan.n_experts}; the written config may not describe the "
              f"pruned checkpoint")
    elif uniform is not None:
        print(f"[surgery] config {', '.join(hits)} -> {uniform}")
    else:
        print(f"[surgery] config {', '.join(hits)} left at {plan.n_experts}: a "
              f"non-uniform plan has no single expert count. Per-layer counts "
              f"are in {SIDECAR}; load with moe_pruner.resize.load_pruned_model")
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))


def prune_checkpoint(plan: PruningPlan, src: Path, dst: Path,
                     dry_run: bool = False) -> dict:
    """
    Write ``src`` to ``dst`` with the plan's dropped experts removed.

    Args:
        plan: The pruning plan; its layer order must match the checkpoint's.
        src: A safetensors checkpoint directory.
        dst: Where to write. Created if missing.
        dry_run: Report what would change and write nothing.

    Returns:
        A summary dict: tensors touched, parameters before and after.
    """
    src, dst = Path(src), Path(dst)
    shapes, weight_map = _read_shapes(src)
    prefixes = _discover_blocks(shapes, plan.n_experts, plan.module_names)
    if len(prefixes) != plan.n_layers:
        raise ValueError(f"the plan covers {plan.n_layers} MoE layers but the "
                         f"checkpoint has {len(prefixes)} ({prefixes[0]} .. "
                         f"{prefixes[-1]}); this is not the checkpoint it was "
                         f"written against")

    # key -> (layer, kind, payload). Built once, then applied shard by shard.
    plans: dict[str, tuple[int, str, object]] = {}
    renames: dict[str, str] = {}
    for l, prefix in enumerate(prefixes):
        keep = plan.keep[l]
        remap = plan.remap(l)
        sliced, listed, _ = _classify(prefix, shapes, plan.n_experts)
        for key in sliced:
            plans[key] = (l, "slice", keep)
        for key, (eid, tail) in listed.items():
            if eid in remap:
                plans[key] = (l, "keep", None)
                renames[key] = f"{prefix}.experts.{remap[eid]}.{tail}"
        if not sliced and not listed:
            raise ValueError(f"found no per-expert tensors under {prefix}")

    def numel(shape):
        n = 1
        for d in shape:
            n *= d
        return n

    before = sum(numel(shapes[k]) for k in plans)
    after = sum(numel([len(plan.keep[l])] + shapes[k][1:]) if kind == "slice"
                else numel(shapes[k])
                for k, (l, kind, _) in plans.items())
    dropped = [k for k in shapes
               if any(k.startswith(p + ".experts.") for p in prefixes)
               and k not in plans]
    summary = {"moe_layers": len(prefixes), "tensors_sliced": len(plans),
               "tensors_dropped": len(dropped),
               "expert_params_before": before + sum(numel(shapes[k]) for k in dropped),
               "expert_params_after": after,
               "n_keep": plan.n_keep.tolist()}

    # Collapse the per-expert index so a ModuleList layout reports one line
    # per weight rather than one per expert.
    kinds = sorted({re.sub(r"(^|\.)experts\.\d+\.", r"\1experts.*.",
                           k[len(prefixes[0]) + 1:]) for k in plans
                    if k.startswith(prefixes[0] + ".")})
    b4 = summary["expert_params_before"]
    print(f"[surgery] {len(prefixes)} MoE blocks, {prefixes[0]} .. {prefixes[-1]}")
    print(f"[surgery] per block: {', '.join(kinds)}")
    print(f"[surgery] routed-expert parameters {b4:,} -> {after:,} "
          f"({100 * after / max(b4, 1):.1f}%); "
          f"{2 * b4 / 1e9:.2f} GB -> {2 * after / 1e9:.2f} GB at bf16")
    if dry_run:
        print("[surgery] dry run; nothing written")
        return summary

    dst.mkdir(parents=True, exist_ok=True)
    new_map: dict[str, str] = {}
    total_bytes = 0
    for fname in sorted(set(weight_map.values())):
        out: dict = {}
        with safe_open(src / fname, framework="pt") as f:
            for key in f.keys():
                entry = plans.get(key)
                if key in dropped:
                    continue
                t = f.get_tensor(key)
                if entry is not None and entry[1] == "slice":
                    t = t[entry[2]].contiguous()
                out[renames.get(key, key)] = t
        if not out:
            continue
        save_file(out, dst / fname, metadata={"format": "pt"})
        for k, t in out.items():
            new_map[k] = fname
            total_bytes += t.numel() * t.element_size()
        print(f"[surgery] {fname}  {len(out)} tensors", flush=True)

    if (src / "model.safetensors.index.json").exists():
        (dst / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": new_map}, indent=2))

    counts = set(plan.n_keep.tolist())
    _update_config(src, dst, plan, counts.pop() if len(counts) == 1 else None)
    plan.write(dst / SIDECAR)

    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name not in (
                "config.json", "model.safetensors.index.json", SIDECAR):
            shutil.copy(f, dst / f.name)
    print(f"[surgery] wrote {dst}")
    return summary
