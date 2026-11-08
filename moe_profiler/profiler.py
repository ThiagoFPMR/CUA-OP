"""MoEProfiler: a loaded MoE model with a `MoEProbe` on every MoE block.

It picks the blocks to hook, fixes the routing settings, and builds the probes.
It does not run generation; `server.generation` does that.
"""
from .discovery import find_moe_blocks
from .probe import MoEProbe
from .store import DEFAULT_TOP_M


def context_window(model) -> int | None:
    """
    The model's positional limit, searched across the nested configs VLMs use.

    Generation is bounded by this and nothing else: the model stops when it
    emits EOS, or when it runs out of positions to attend over. Returns None
    when the checkpoint does not declare a limit.
    """
    cfgs = [model.config]
    cfgs += [c for c in (getattr(model.config, a, None)
                         for a in ("text_config", "llm_config", "language_config"))
             if c is not None]
    limits = [v for v in (getattr(c, "max_position_embeddings", None) for c in cfgs)
              if isinstance(v, int) and v > 0]
    return max(limits) if limits else None


class MoEProfiler:
    """
    A loaded MoE model, the blocks to probe, and one probe per block.

    The model, the hooked blocks and every routing hyperparameter are fixed at
    construction. The probes persist across generations: their routing/expert
    adapters are resolved on the first forward and reused, and the fused
    reconstruction cross-check runs only on the very first one.
    """

    def __init__(
        self,
        model,
        processor,
        top_k: int,
        scoring_func: str,
        norm_topk_prob: bool,
        scope: str | None,
        verify: bool,
        meta_extra: dict | None = None,
        do_sample: bool = False,
        top_m: int = DEFAULT_TOP_M,
    ):
        """
        Args:
            model: The loaded MoE model.
            processor: The multimodal processor for encoding inputs.
            top_k: Number of experts to select per token.
            scoring_func: Scoring function to use ('softmax' or 'sigmoid').
            norm_topk_prob: Whether to normalize the top-k probabilities.
            scope: Optional regex to select which MoE blocks to hook.
            verify: Whether to perform verification of fused expert reconstruction.
            meta_extra: Model-level metadata (e.g. model path and type) for the
                caller to record alongside the stats.
            do_sample: Sample instead of decoding greedily. Greedy by default so
                a rerun reproduces both the text and the routing exactly.
            top_m: Expert columns kept per token in the routing stats. Held
                to at least top_k, which is the whole selection; anything
                beyond that is the near-miss margin.
        """
        self.model = model
        self.processor = processor
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.scope = scope
        self.verify = verify
        self.meta_extra = dict(meta_extra or {})
        self.do_sample = do_sample
        self.top_m = max(top_m, top_k or 0)
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.context_window = context_window(model)
        if self.context_window is None:
            print("[warn] the checkpoint declares no max_position_embeddings; "
                  "generation length falls back to the model's own "
                  "generation_config and may be clipped", flush=True)
        else:
            print(f"[cfg] context_window={self.context_window}", flush=True)

        self.blocks = find_moe_blocks(model, scope)
        self.device = next(model.parameters()).device
        self.layer_indices = [l for _, l, _ in self.blocks]
        self.module_names = [n for n, _, _ in self.blocks]
        self.probes = [MoEProbe(m, top_k, scoring_func, norm_topk_prob, verify,
                                top_m=self.top_m)
                       for _, _, m in self.blocks]
