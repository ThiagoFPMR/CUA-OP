"""Time and check the profiling path on a real checkpoint, outside the server.

Run it on the GPU box with the server stopped. --verify checks the probes against
the reference probe; --compare_paths measures how much the server's decoding
changes the recorded routing.
"""
from __future__ import annotations

import argparse
import logging
import statistics
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteriaList

from ..encoding import encode_messages
from ..reference_probe import ReferenceProbe
from ..store import pack_generation

from .app import ServerConfig, build_profiler
from .decode_graph import DecodeGraph
from .generation import (_TokenClock, _peak_gib, _reset_peaks, _vision_tower,
                         profile_generation, register_probes,
                         release_cuda_memory)

logger = logging.getLogger("moe_profiler.bench")

MODES = ("model", "model-graph", "fast", "fast-graph", "reference")

FILLER = ("Previous step: the agent clicked the toolbar, waited for the dialog to "
          "open, and read the values shown in the spreadsheet cells. ")


def _screenshot(width: int, height: int, seed: int) -> Image.Image:
    """A desktop-like image: panels, bars and text. Only its size decides how
    many tokens it becomes; the content just gives the model something to read."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (width, height), (238, 238, 238))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, 36], fill=(52, 58, 64))
    for _ in range(60):
        x, y = int(rng.integers(0, width - 200)), int(rng.integers(40, height - 40))
        w, h = int(rng.integers(60, 400)), int(rng.integers(16, 120))
        draw.rectangle([x, y, x + w, y + h], outline=(90, 90, 90),
                       fill=tuple(int(c) for c in rng.integers(180, 255, 3)))
        draw.text((x + 4, y + 2), f"Cell {int(rng.integers(1, 999))} = "
                  f"{rng.random() * 1000:.2f}", fill=(20, 20, 20))
    return img


def build_prompt(profiler, n_images: int, width: int, height: int,
                 prompt_tokens: int, new_tokens: int, screenshots=None) -> dict:
    """A prompt of about ``prompt_tokens`` input ids, answering exactly
    ``new_tokens`` tokens: ``n_images`` screenshots, then text history. Real
    ``screenshots`` (paths) are used instead of synthetic ones when given."""
    images = ([Image.open(p).convert("RGB") for p in screenshots] if screenshots
              else [_screenshot(width, height, seed=i) for i in range(n_images)])
    instruction = ("You are operating a computer. Look at the screenshots and "
                   "decide the next action. Explain your reasoning, then give "
                   "the action.\n")

    def prompt(n_filler: int) -> dict:
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": FILLER * n_filler + instruction})
        return {"messages": [{"role": "user", "content": content}],
                "images": images, "max_new_tokens": new_tokens,
                "min_new_tokens": new_tokens}

    def length(p: dict) -> int:
        enc, _ = encode_messages(profiler.processor, p["messages"], p["images"],
                                    "cpu", None)
        return enc["input_ids"].shape[1]

    base = length(prompt(0))
    per = max(1, length(prompt(8)) - base) / 8
    return prompt(max(0, round((prompt_tokens - base) / per)))


def _graph(profiler, probes: list, budget: int, replay: bool = True) -> DecodeGraph:
    """A `DecodeGraph` whose cache is laid out as the server's would be: for the
    whole answer budget an agent asks for, not just the tokens a benchmark pins."""
    graph = DecodeGraph(profiler.model, probes, replay=replay)
    start = graph.start
    graph.start = lambda n_prompt, max_new_tokens: start(n_prompt, budget)
    return graph


def _generate(profiler, prompt: dict, graph: DecodeGraph | None) -> tuple:
    """`generate` on the prompt as the server would run it, without probes."""
    enc, _ = encode_messages(profiler.processor, prompt["messages"],
                                prompt["images"], profiler.device, None)
    clock = _TokenClock(profiler.probes)
    kwargs = {"do_sample": False, "num_beams": 1, "use_cache": True,
              "max_new_tokens": prompt["max_new_tokens"],
              "min_new_tokens": prompt["min_new_tokens"],
              "stopping_criteria": StoppingCriteriaList([clock])}
    if graph is not None:
        kwargs["past_key_values"] = graph.start(enc["input_ids"].shape[1],
                                                prompt["max_new_tokens"])
    try:
        sequences = profiler.model.generate(**enc, **kwargs)
    finally:
        steps = graph.end() if graph is not None else {}
    return sequences, {**clock.timings(), **steps}, enc["input_ids"].shape[1]


def run_mode(profiler, prompt: dict, mode: str, budget: int) -> tuple[dict, dict | None]:
    """One timed generation in ``mode``. Returns its timings, and the packed
    stats when probes ran."""
    release_cuda_memory()
    _reset_peaks()
    graphed = mode.endswith("-graph")
    started = time.perf_counter()
    arrays = None
    if mode.startswith("fast"):
        handles = register_probes(profiler)
        graph = _graph(profiler, profiler.probes, budget) if graphed else None
        try:
            result = profile_generation(profiler, prompt, decode_graph=graph)
        finally:
            if graph is not None:
                graph.uninstall()
            for handle in handles:
                handle.remove()
        t, arrays = dict(result.timings), result.arrays
        n_prompt = result.n_prompt_input_ids
        n_new = len(result.generated_token_ids)
    else:
        refs, handles = [], []
        if mode == "reference":
            refs = [ReferenceProbe(module, profiler.top_k, profiler.scoring_func,
                                   profiler.norm_topk_prob, verify=False,
                                   diagnostics=True)
                    for _, _, module in profiler.blocks]
            handles = [module.register_forward_hook(r.hook, with_kwargs=True)
                       for (_, _, module), r in zip(profiler.blocks, refs)]
        graph = _graph(profiler, [], budget) if graphed else None
        try:
            with torch.inference_mode():
                sequences, t, n_prompt = _generate(profiler, prompt, graph)
        finally:
            if graph is not None:
                graph.uninstall()
            for handle in handles:
                handle.remove()
        n_new = sequences.shape[1] - n_prompt
        if refs:
            packing = time.perf_counter()
            for r in refs:
                r.finalize()
            arrays = _pack_reference(refs, profiler.top_m)
            t["pack_s"] = time.perf_counter() - packing
    t.update(total_s=time.perf_counter() - started, n_prompt=n_prompt,
             n_new=n_new, peak_gib=_peak_gib())
    return t, arrays


def _pack_reference(refs: list, top_m: int) -> dict:
    return pack_generation(np.stack([r.scores.numpy() for r in refs]),
                           np.stack([r.norms.numpy() for r in refs]),
                           np.stack([r.sim.numpy() for r in refs]),
                           n_experts=refs[0].n_experts, top_m=top_m,
                           scattered=bool(refs[0].scores_scattered))


def time_vision(profiler, prompt: dict) -> None:
    """The vision tower's share of the prefill, on this prompt's screenshots."""
    tower = _vision_tower(profiler.model)
    if tower is None or not prompt["images"]:
        return
    enc, _ = encode_messages(profiler.processor, prompt["messages"],
                                prompt["images"], profiler.device, None)
    pixels = enc["pixel_values"].to(next(tower.parameters()).dtype)
    with torch.inference_mode():
        for _ in range(2):          # the first call warms up
            torch.cuda.synchronize()
            started = time.perf_counter()
            tower(pixels, grid_thw=enc["image_grid_thw"])
            torch.cuda.synchronize()
    print(f"\n== vision tower: {time.perf_counter() - started:.2f}s for "
          f"{len(prompt['images'])} images (part of every prefill)")


def _compare(label: str, a: dict, b: dict, tolerance: float | None) -> bool:
    """Print how two packed generations differ. ``tolerance`` bounds the p99.9
    relative error of norms/impact; None demands bit-identical arrays."""
    for key in a:
        if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype:
            print(f"  {label}: {key} shape/dtype MISMATCH {a[key].shape}/{a[key].dtype} "
                  f"vs {b[key].shape}/{b[key].dtype}")
            return False
    if tolerance is None:
        same = all(np.array_equal(a[k], b[k]) for k in a)
        print(f"  {label}: {'bit-identical' if same else 'NOT bit-identical'}")
        return same
    ok = True
    for key in ("norms", "impact"):
        x, y = a[key].astype(np.float64), b[key].astype(np.float64)
        rel = np.abs(x - y) / np.maximum(np.abs(x), 1e-12)
        good = np.quantile(rel, 0.999) <= tolerance
        print(f"  {label}: {key} {np.mean(x == y):.2%} bit-identical, rel err "
              f"median {np.median(rel):.1e} p99.9 {np.quantile(rel, 0.999):.1e} "
              f"max {rel.max():.1e}{'' if good else '  <-- LARGER THAN EXPECTED'}")
        ok &= bool(good)
    return ok


def verify(profiler, prompt: dict, budget: int) -> bool:
    """Do the fast probe and the replayed decode steps store what they should?"""
    ok = True
    probe = profiler.probes[0]

    # 1. Original vs fast probe, both eager, on one generation.
    refs = [ReferenceProbe(module, profiler.top_k, profiler.scoring_func,
                           profiler.norm_topk_prob, verify=False, diagnostics=True)
            for _, _, module in profiler.blocks]
    # The reference hooks go on first, so they run first in each block.
    handles = [module.register_forward_hook(r.hook, with_kwargs=True)
               for (_, _, module), r in zip(profiler.blocks, refs)]
    handles += register_probes(profiler)
    probe.verify, probe.verdict = True, None
    try:
        result = profile_generation(profiler, prompt)
    finally:
        for handle in handles:
            handle.remove()
    for r in refs:
        r.finalize()
    a, b = _pack_reference(refs, profiler.top_m), result.arrays
    print(f"\n== verify: {len(refs)} blocks, {result.n_activation_rows} token rows, "
          f"experts via {'grouped GEMM' if probe.grouped_fn is not None else 'per-expert fallback'}")
    print(f"  fused reconstruction vs the model's experts: {probe.verdict}")
    same_logits = np.array_equal(a["top_logit"], b["top_logit"])
    print(f"  original vs fast probe: top_logit bit-identical: {same_logits}")
    ok &= same_logits
    moved = a["top_idx"] != b["top_idx"]
    if moved.any():
        # With the logits identical, a differing id sits where the reference
        # had the same logit too: a tie, which the two orders break differently.
        print(f"  original vs fast probe: {moved.sum()} of {moved.size} expert ids "
              f"differ, all at tied logits; the fast probe stores the router's "
              f"own pick there")
    ok &= _compare("original vs fast probe", a, b, tolerance=2e-3)

    # 2. Replayed decode steps vs the same static-cache steps run eagerly.
    runs = {}
    for label, replay in (("static eager", False), ("static replayed", True)):
        handles = register_probes(profiler)
        graph = _graph(profiler, profiler.probes, budget, replay=replay)
        try:
            runs[label] = profile_generation(profiler, prompt, decode_graph=graph)
        finally:
            graph.uninstall()
            for handle in handles:
                handle.remove()
    replayed = runs["static replayed"]
    print(f"  decode steps replayed from the graph: "
          f"{replayed.timings.get('graph_replays', 0)} of "
          f"{len(replayed.generated_token_ids) - 1}")
    ok &= replayed.timings.get("graph_replays", 0) > 0
    same_text = runs["static eager"].generated_token_ids == replayed.generated_token_ids
    print(f"  replayed vs eager (static cache): same tokens: {same_text}")
    ok &= same_text
    ok &= _compare("replayed vs eager (static cache)", runs["static eager"].arrays,
                   replayed.arrays, tolerance=None)

    # 3. What the static cache itself changes (not a pass/fail: kernels differ).
    same_text = result.generated_token_ids == replayed.generated_token_ids
    print(f"  static vs default cache (informational): same tokens: {same_text}")
    _compare("static vs default cache (informational)", b, replayed.arrays,
             tolerance=float("inf"))

    # 4. One decode-step probe call with the device told to fail on any host wait.
    if torch.cuda.is_available():
        block = profiler.blocks[0][2]
        hidden = profiler.model.get_input_embeddings().weight.shape[1]
        h = torch.randn(1, 1, hidden, device=profiler.device, dtype=torch.bfloat16)
        router = getattr(block, probe.router_attr)
        with torch.inference_mode():
            for attempt in range(2):          # the first call warms up
                probe.reset()
                probe._router_out = router(h.reshape(-1, hidden))
                torch.cuda.synchronize()
                if attempt:
                    torch.cuda.set_sync_debug_mode("error")
                try:
                    probe.hook(block, (h,), {}, None)
                    if attempt:
                        print("  decode-step probe call: never waits on the device")
                except RuntimeError as exc:
                    print(f"  decode-step probe call WAITS ON THE DEVICE: {exc}")
                    if torch.cuda.get_device_capability(profiler.device)[0] < 8:
                        print("    (expected below SM 80: torch runs grouped GEMMs "
                              "through a fallback that reads the group sizes back "
                              "to the host; SM 80+ runs them natively)")
                    ok = False
                finally:
                    torch.cuda.set_sync_debug_mode(0)
        probe.reset()
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


class _Force(LogitsProcessor):
    """Makes `generate` answer with these tokens, whatever the model scores."""

    def __init__(self, tokens: list):
        self.tokens, self.step = tokens, 0

    def __call__(self, input_ids, scores):
        forced = torch.full_like(scores, float("-inf"))
        forced[:, self.tokens[self.step]] = 0
        self.step += 1
        return forced


def _jsd_bits(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2
    kl = lambda a: float(np.sum(a[a > 0] * np.log2(a[a > 0] / m[a > 0])))
    return (kl(p) + kl(q)) / 2


def compare_paths(profiler, prompt: dict, budget: int) -> None:
    """How far does the decode graph move the routing an analysis reads?"""
    def run(graph_on: bool, force=None):
        handles = register_probes(profiler)
        graph = _graph(profiler, profiler.probes, budget) if graph_on else None
        p = prompt if force is None else dict(
            prompt, logits_processor=LogitsProcessorList([_Force(force)]))
        try:
            return profile_generation(profiler, p, decode_graph=graph)
        finally:
            if graph is not None:
                graph.uninstall()
            for handle in handles:
                handle.remove()

    before = run(False)                 # as served before the decode graphs
    tokens = before.generated_token_ids
    forced = run(True, force=tokens)    # as served now, on the same tokens
    free = run(True)                    # as served now, left to itself
    assert forced.generated_token_ids == tokens
    if forced.timings.get("graph_replays", 0) == 0:
        print("  (the decode graph never replayed: this compares eager paths)")

    k, n_prefill = profiler.top_k, before.n_prefill_tokens
    a = np.sort(before.arrays["top_idx"][..., :k].astype(np.int64), -1)
    b = np.sort(forced.arrays["top_idx"][..., :k].astype(np.int64), -1)
    n_experts = int(profiler.probes[0].n_experts)
    first = next((i for i, (x, y) in enumerate(zip(tokens, free.generated_token_ids))
                  if x != y), None)
    print(f"\n== decode path vs as served before: {len(tokens)}-token answer "
          f"to a {n_prefill}-token prompt, {a.shape[0]} MoE layers")
    print("  left to itself, the graph path answers "
          + ("the same" if first is None and len(tokens) == len(free.generated_token_ids)
             else f"differently from token {first if first is not None else min(len(tokens), len(free.generated_token_ids))}"))
    for label, rows in (("prompt", slice(0, n_prefill)), ("answer", slice(n_prefill, None))):
        x, y = a[:, rows], b[:, rows]
        if x.shape[1] == 0:
            continue
        moved = np.any(x != y, axis=-1)                       # [L, T]
        jsd = []
        for layer in range(x.shape[0]):
            p = np.bincount(x[layer].ravel(), minlength=n_experts).astype(np.float64)
            q = np.bincount(y[layer].ravel(), minlength=n_experts).astype(np.float64)
            jsd.append(_jsd_bits(p / p.sum(), q / q.sum()))
        worst = int(np.argmax(jsd))
        print(f"  {label} rows ({x.shape[1]}): top-k set differs in {moved.mean():.3%} "
              f"of (layer, token) pairs (worst layer {moved.mean(1).max():.2%}); "
              f"selection-frequency JSD per layer median {np.median(jsd):.1e} bits, "
              f"max {jsd[worst]:.1e} (layer {worst})")
    _compare("norms/impact, same tokens", before.arrays, forced.arrays,
             tolerance=float("inf"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--experts", default=None,
                    help="the model's experts implementation (grouped_mm, "
                         "batched_mm, eager); default: transformers' choice")
    ap.add_argument("--modes", default=None,
                    help=f"comma-separated, of: {', '.join(MODES)}; default "
                         f"model,model-graph,fast,fast-graph, or none with "
                         f"--compare_paths")
    ap.add_argument("--images", type=int, default=3,
                    help="screenshots per prompt (live steps carry up to 3)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--prompt_tokens", type=int, default=7400,
                    help="target prompt length; the live median is ~7450")
    ap.add_argument("--new_tokens", type=int, default=256,
                    help="answer length; the live median is ~256")
    ap.add_argument("--budget", type=int, default=32768,
                    help="answer budget the static cache is laid out for; the "
                         "agents ask for 32768")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--compare_paths", action="store_true",
                    help="measure how far the decode graph moves the stored routing")
    ap.add_argument("--screenshots", nargs="*", default=None,
                    help="real screenshots (image paths) to prompt with")
    ap.add_argument("--generations", type=int, default=1500,
                    help="run size to project the timings onto")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    if args.modes is None:
        args.modes = "" if args.compare_paths else "model,model-graph,fast,fast-graph"
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if unknown := set(modes) - set(MODES):
        ap.error(f"unknown modes: {sorted(unknown)}")

    profiler = build_profiler(ServerConfig(attn=args.attn), args.model_path)
    if args.experts:
        profiler.model.set_experts_implementation(args.experts)
    prompt = build_prompt(profiler, args.images, args.width, args.height,
                          args.prompt_tokens, args.new_tokens, args.screenshots)

    # Warm-up: CUDA context, cuBLAS handles, any kernels compiled on first use.
    run_mode(profiler, dict(prompt, max_new_tokens=4, min_new_tokens=4), "fast",
             args.budget)
    time_vision(profiler, prompt)

    ok = True
    if args.verify:
        ok = verify(profiler, dict(prompt, max_new_tokens=32, min_new_tokens=32),
                    args.budget)

    if args.compare_paths:
        compare_paths(profiler, prompt, args.budget)

    rows = {}
    for mode in modes:
        runs = [run_mode(profiler, prompt, mode, args.budget)[0]
                for _ in range(args.repeats)]
        rows[mode] = {k: statistics.median(r[k] for r in runs)
                      for k in runs[0] if isinstance(runs[0][k], (int, float))}

    if not rows:
        return 0 if ok else 1
    print(f"\n== timings: median of {args.repeats}; {args.images} images "
          f"{args.width}x{args.height}; answer budget {args.budget} tokens")
    print(f"{'mode':12s} {'prompt':>7s} {'new':>5s} {'prefill s':>10s} "
          f"{'decode s':>9s} {'ms/token':>9s} {'replayed':>9s} "
          f"{'total s':>8s} {'peak GiB':>9s} {'run of ' + str(args.generations):>12s}")
    for mode, t in rows.items():
        steps = max(t.get("decode_steps", 1), 1)
        hours = t["total_s"] * args.generations / 3600
        replayed = (f"{t['graph_replays']:.0f}/{steps:.0f}" if "graph_replays" in t
                    else "-")
        if t.get("graph_growths"):
            replayed += f"+{t['graph_growths']:.0f}g"
        print(f"{mode:12s} {t['n_prompt']:7.0f} {t['n_new']:5.0f} "
              f"{t.get('prefill_s', 0):10.2f} {t.get('decode_s', 0):9.2f} "
              f"{1000 * t.get('decode_s', 0) / steps:9.1f} {replayed:>9s} "
              f"{t['total_s']:8.2f} {t['peak_gib']:9.1f} {hours:10.1f} h")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
