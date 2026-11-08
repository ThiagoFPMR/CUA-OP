"""Check a pruned checkpoint against the original model with the same plan masked on.

One forward pass on random tokens compares their logits. They should match, which
confirms the surgery sliced the right tensors.
"""
import torch

from .masking import pruned
from .plan import PruningPlan


@torch.inference_mode()
def compare_logits(full_model, pruned_model, plan: PruningPlan, inputs) -> dict:
    """
    Run both models on the same inputs and report how far apart they land.

    Args:
        full_model: The unpruned model, to be masked with the plan.
        pruned_model: The model loaded from the written checkpoint.
        plan: The plan both are meant to implement.
        inputs: A kwargs dict to call both models with.

    Returns:
        ``{"max_abs", "rel", "argmax_agreement"}``.
    """
    with pruned(full_model, plan):
        a = full_model(**inputs).logits.float()
    b = pruned_model(**{k: v.to(next(pruned_model.parameters()).device)
                        if torch.is_tensor(v) else v
                        for k, v in inputs.items()}).logits.float().to(a.device)
    diff = (a - b).abs()
    return {"max_abs": diff.max().item(),
            "rel": (diff.max() / a.abs().max().clamp_min(1e-9)).item(),
            "argmax_agreement": (a.argmax(-1) == b.argmax(-1)).float().mean().item()}


def verify_checkpoint(src, dst, plan: PruningPlan, n_tokens: int = 32,
                      dtype=torch.bfloat16, device_map: str | None = "auto") -> dict:
    """Load both checkpoints and compare them on random token ids."""
    from .resize import _try_factories, load_pruned_model

    full = _try_factories(
        lambda f: f.from_pretrained(str(src), dtype=dtype, device_map=device_map,
                                    trust_remote_code=True), "load the original")
    full.eval()
    small, _ = load_pruned_model(dst, plan=plan, dtype=dtype, device_map=device_map)
    cfg = full.config
    vocab = getattr(cfg, "vocab_size", None) or getattr(
        getattr(cfg, "text_config", cfg), "vocab_size")
    ids = torch.randint(0, vocab, (1, n_tokens), device=next(full.parameters()).device)
    out = compare_logits(full, small, plan, {"input_ids": ids})
    print(f"[verify] max|masked - pruned| = {out['max_abs']:.3e} "
          f"(rel {out['rel']:.3e}), argmax agreement "
          f"{out['argmax_agreement']:.4f}")
    return out
