#!/usr/bin/env python3
"""Prune a MoE checkpoint from profiled episodes.

Scores every expert over the runs, decides how many each layer keeps, writes the
plan, and, given --ckpt and --dst, cuts the checkpoint:

    uv run scripts/prune.py /data/runs/my-ckpt --fraction 0.5 --out plan.json
    uv run scripts/prune.py --plan plan.json --ckpt /models/my-ckpt --dst /models/my-ckpt-50
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from moe_profiler.run import Run
from moe_pruner.budget import WEIGHTINGS, allocate, curve_points, tradeoff, uniform_curve
from moe_pruner.plan import PruningPlan
from moe_pruner.scoring import DEFAULT_CRITERION, PRESETS, ExpertScores
from moe_pruner.surgery import prune_checkpoint

META = "run.meta.json"


def find_runs(paths: list[Path], domains: set[str], stop_reasons: set[str]) -> list[Run]:
    """Every run under `paths` that passes the filters and has generations."""
    metas: list[Path] = []
    for p in paths:
        if p.is_file():
            metas.append(p)
        elif (p / META).is_file():
            metas.append(p / META)
        elif p.is_dir():
            metas.extend(sorted(p.rglob(META)))
        else:
            raise SystemExit(f"no such run or directory: {p}")

    runs, skipped = [], {"by domain": 0, "by stop reason": 0, "empty": 0}
    # Resolved, so the plan's provenance names each run absolutely; deduped in order.
    for path in dict.fromkeys(p.resolve() for p in metas):
        meta = json.loads(path.read_text())
        if domains and meta.get("domain") not in domains:
            skipped["by domain"] += 1
        elif stop_reasons and meta.get("stop_reason") not in stop_reasons:
            skipped["by stop reason"] += 1
        elif not meta.get("generations_written"):
            skipped["empty"] += 1
        else:
            runs.append(Run.open(path))
    left_out = ", ".join(f"{n} {why}" for why, n in skipped.items() if n)
    print(f"[runs] {len(runs)} runs" + (f" ({left_out} left out)" if left_out else ""))
    if not runs:
        raise SystemExit("no runs left to score")
    return runs


def criterion_spec(args) -> dict:
    """The criterion fields the command line asked for, and only those."""
    spec = dataclasses.asdict(PRESETS[args.criterion]) if args.criterion else {}
    return spec | {k: getattr(args, k) for k in ("b", "alpha", "beta", "gamma")
                   if getattr(args, k) is not None}


def load_scores(args) -> ExpertScores:
    """The accumulators, from a --scores cache or a pass over the runs."""
    cached = ExpertScores.load(args.scores) if args.scores else None
    crit = dataclasses.replace(cached.criterion if cached else DEFAULT_CRITERION,
                               **criterion_spec(args))
    if crit.degenerate:
        raise SystemExit(f"{crit} scores every routed expert 1, which ranks nothing; "
                         f"give it a gate, activation or token exponent")
    if cached:
        print(f"[scores] {args.scores}  {crit}")
        return cached.with_criterion(crit)

    runs = find_runs(args.runs, set(args.domain), set(args.stop_reason))
    print(f"[score] {crit}")
    scores = ExpertScores.from_runs(runs, criterion=crit)
    if args.save_scores:
        scores.save(args.save_scores)
        print(f"[scores] cached to {args.save_scores}")
    return scores


def report(alloc, scores: ExpertScores, min_experts: int) -> None:
    """What the allocation did, and what a uniform budget would have bought."""
    S = scores.importance()
    L, E = S.shape
    uni = uniform_curve(alloc.retention, alloc.weight)
    kept = alloc.total_kept
    print(f"\n{'mode':<16}{'experts kept':>14}{'% of all':>10}{'retention':>12}")
    print(f"{alloc.mode + '/' + alloc.weighting:<16}{kept:>14,}"
          f"{100 * kept / (L * E):>9.1f}%{alloc.objective:>12.4f}")
    if alloc.mode != "uniform":
        n = round(kept / L)
        if 1 <= n <= E:
            print(f"{'uniform':<16}{n * L:>14,}{100 * n / E:>9.1f}%{uni[n - 1]:>12.4f}"
                  f"   <- same budget, spread evenly")
        reach = np.flatnonzero(uni >= alloc.objective)
        if len(reach):
            n = int(reach[0]) + 1
            print(f"{'uniform':<16}{n * L:>14,}{100 * n / E:>9.1f}%{uni[n - 1]:>12.4f}"
                  f"   <- same retention, spread evenly")
    print("(retention is the share of weighted importance kept; it compares "
          "allocations of the same weighting only)")
    print(f"\nper-layer budget ({alloc.n_keep.min()}-{alloc.n_keep.max()} experts, "
          f"floor {min_experts}):")
    print("  " + " ".join(f"L{l}:{alloc.n_keep[l]}" for l in range(L)))
    dead = int((S.sum(1) <= 0).sum())
    if dead:
        print(f"[warn] {dead} layer(s) were never routed to in the scored "
              f"generations; they keep the floor and nothing more")


def build_plan(args) -> PruningPlan:
    """Score the runs (or read the cache) and allocate the budget."""
    scores = load_scores(args)
    importance, impact = scores.importance(), scores.layer_impact()
    meta = scores.meta
    top_k = meta["top_k"]
    min_experts = top_k if args.min_experts is None else args.min_experts

    alloc = allocate(importance, impact, mode=args.mode, weighting=args.weight,
                     total=args.total, per_layer=args.per_layer,
                     fraction=args.fraction, target=args.retention,
                     min_experts=min_experts, max_experts=args.max_experts)
    requested = {k: v for k, v in (("per_layer", args.per_layer), ("total", args.total),
                                   ("fraction", args.fraction),
                                   ("retention", args.retention)) if v is not None}
    plan = PruningPlan.from_allocation(
        alloc, n_experts=scores.n_experts, top_k=top_k,
        layer_indices=meta.get("layer_indices"), module_names=meta.get("module_names"),
        source={"model_path": meta.get("model_path"),
                "model_type": meta.get("model_type"),
                "criterion": scores.criterion.as_dict(),
                "n_generations": scores.n_generations,
                "runs": meta.get("runs", [])},
        constraints={"min_experts": min_experts, "max_experts": args.max_experts,
                     "requested": requested},
        extra_diagnostics={"per_layer_block_impact": [float(x) for x in impact],
                           # So another budget can be read off without rescoring.
                           "retention_tradeoff": tradeoff(alloc),
                           "budget_curve": curve_points(alloc)})
    report(alloc, scores, min_experts)
    return plan


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="*",
                    help="run.meta.json files, episode directories, or directories "
                         "searched recursively for them")

    src = ap.add_argument_group("where the expert scores come from")
    pick = src.add_mutually_exclusive_group()
    pick.add_argument("--scores", type=Path, help="a cache written by --save-scores")
    pick.add_argument("--plan", type=Path, help="skip scoring and apply this plan")
    src.add_argument("--save-scores", type=Path,
                     help="cache the accumulators here for later allocations")
    src.add_argument("--domain", action="append", default=[],
                     help="score only runs of this domain (repeatable)")
    src.add_argument("--stop-reason", action="append", default=[],
                     help="score only runs that ended this way, e.g. complete or "
                          "max_steps (repeatable)")

    sc = ap.add_argument_group("scoring")
    sc.add_argument("--criterion", choices=tuple(PRESETS), default=None,
                    help="a published criterion to start from (default: reap, "
                         "or a --scores cache's own)")
    for name, what in (("b", "routing-frequency normalization: divide by N_i^b"),
                       ("alpha", "gate-weight exponent"),
                       ("beta", "expert-output-norm exponent"),
                       ("gamma", "token-contribution (1 - cos) exponent")):
        sc.add_argument(f"--{name}", type=float, default=None,
                        help=f"{what}; overrides --criterion")

    bu = ap.add_argument_group("budget")
    bu.add_argument("--mode", choices=("greedy", "uniform"), default="greedy",
                    help="greedy spends the budget where the residual moves; uniform "
                         "keeps one count everywhere, which serving stacks expect")
    bu.add_argument("--weight", choices=WEIGHTINGS, default="impact",
                    help="what the greedy allocates by (default: %(default)s)")
    size = bu.add_mutually_exclusive_group()
    size.add_argument("--per-layer", type=int, help="experts per layer on average")
    size.add_argument("--total", type=int, help="experts kept across the whole stack")
    size.add_argument("--fraction", type=float, help="share of all L x E experts")
    size.add_argument("--retention", type=float, metavar="R",
                      help="weighted share of importance to keep (e.g. 0.95); the "
                           "budget is whatever that takes")
    bu.add_argument("--min-experts", type=int, default=None,
                    help="floor per layer (default: top-k, below which the router "
                         "cannot fill its selection)")
    bu.add_argument("--max-experts", type=int, default=None,
                    help="ceiling per layer (default: E)")

    out = ap.add_argument_group("output")
    out.add_argument("--out", type=Path, default=Path("expert_plan.json"),
                     help="where to write the plan (default: %(default)s)")
    out.add_argument("--mask-json", type=Path, default=None,
                     help="also export EASY-EP's [L][E] 0/1 mask matrix")

    cut = ap.add_argument_group("cutting the checkpoint (omit to stop at the plan)")
    cut.add_argument("--ckpt", type=Path, help="the safetensors checkpoint to prune")
    cut.add_argument("--dst", type=Path, help="where to write the pruned one")
    cut.add_argument("--dry-run", action="store_true",
                     help="report what the surgery would change and write no weights")
    cut.add_argument("--verify", action="store_true",
                     help="after cutting, load the pruned checkpoint and the masked "
                          "original and check they agree (needs memory for both)")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    if bool(args.ckpt) != bool(args.dst):
        ap.error("--ckpt and --dst go together; omit both to stop at the plan")
    if args.plan:
        if args.runs:
            ap.error("--plan replaces scoring runs; give one or the other")
        if not args.dst:
            ap.error("--plan needs --ckpt/--dst; without them there is nothing to do")
        plan = PruningPlan.read(args.plan)
    else:
        if bool(args.runs) == bool(args.scores):
            ap.error("give runs to score, or a --scores cache (not both)")
        if all(v is None for v in (args.per_layer, args.total, args.fraction,
                                   args.retention)):
            ap.error("say how big the budget is: one of --per-layer / --total / "
                     "--fraction / --retention")
        plan = build_plan(args)
        plan.write(args.out)
        print(f"\n[plan] {args.out}")
    if args.mask_json:
        plan.write_mask_json(args.mask_json)
        print(f"[mask] {args.mask_json}")

    if not args.dst:
        print("[done] no --dst, so nothing was cut. Try the plan first with "
              "moe_pruner.pruned(model, plan), then pass --ckpt/--dst.")
        return 0

    counts = sorted(set(plan.n_keep.tolist()))
    shape = (f"uniform at {counts[0]}" if len(counts) == 1
             else f"non-uniform {counts[0]}-{counts[-1]}")
    print(f"\n[cut] {plan.n_keep.sum():,} of {plan.n_layers * plan.n_experts:,} "
          f"experts, {shape}")
    if plan.n_keep.min() < plan.top_k:
        print(f"[cut] WARNING: a layer keeps {plan.n_keep.min()} experts but the "
              f"router selects top-{plan.top_k}")
    prune_checkpoint(plan, args.ckpt, args.dst, dry_run=args.dry_run)
    if args.verify and not args.dry_run:
        from moe_pruner.verify import verify_checkpoint
        verify_checkpoint(args.ckpt, args.dst, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
