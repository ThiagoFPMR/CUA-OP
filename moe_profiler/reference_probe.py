"""ReferenceProbe: a slow, straightforward MoE probe, kept to check `MoEProbe` against.

It records the same per-token stats as `probe.MoEProbe`, but one expert at a time
and waiting on the GPU at each step, which is far too slow to profile with.
`moe-profiler-bench --verify` checks that both probes store the same values.
"""
import torch
from torch import nn
import torch.nn.functional as F

from .discovery import router_of
from .experts import build_expert_fn, verify_fused
from .routing import build_route_fn


class ReferenceProbe:
    """
    One instance per MoE block; a forward hook fills per-token routing stats.

    The probe is initialized with the MoE block and the parameters for the routing
    and expert functions. It is then attached as a forward hook to the block.
    """

    def __init__(self, 
        block: nn.Module, 
        top_k: int, 
        scoring_func: str, 
        norm_topk_prob: bool, 
        verify: bool = True,
        diagnostics: bool = False,
    ):
        """
        Initialize a probe for one MoE block.

        Args:
            block: The MoE block module to probe.
            top_k: The number of top experts to consider for routing.
            scoring_func: The scoring function to use for routing.
            norm_topk_prob: Whether to normalize the top-k probabilities.
            verify: Whether to verify the fused reconstruction against experts' forward.
            diagnostics: Whether to collect additional diagnostic information.
        """
        self.block = block
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.verify = verify
        self.diagnostics = diagnostics
        self.route_fn = None
        self.expert_fn = None
        self.n_experts = None
        self.layout = None
        self.router_attr = None
        self.verdict = None
        self.scores_scattered = None  # True if full_activation_scores is a top-k scatter, not
        self.reset()                  # the router's own full distribution

    def reset(self):
        """Reset the probe's recorded stats. Call this before each demonstration."""
        self.top_k_idxs = None      # [T, k] long
        self.top_exp_weights = None   # [T, k] float
        self.norms = None     # [T, k] float
        self.sim = None       # [T] float
        self.scores = None    # [T, E] float16 -- full router distribution
        self._chunks = []     # one entry per forward, fields as in `finalize`
        self.chunk_sizes = [] # tokens seen per forward; [0] is the prefill length

    def finalize(self):
        """
        Concatenate the per-forward chunks into the [T, ...] per-demonstration
        stats. Call once after `generate` returns, before reading the stats.
        Leaves everything None when the block never fired.
        """
        if not self._chunks:
            return
        idxs, weights, norms, sim, scores = zip(*self._chunks)
        self.top_k_idxs = torch.cat(idxs)
        self.top_exp_weights = torch.cat(weights)
        self.norms = torch.cat(norms)
        self.sim = torch.cat(sim)
        self.scores = torch.cat(scores) if self.diagnostics else None

    def _build(self, hidden_size: int):
        """
        Build the routing and expert adapters on the first forward (when the
        true hidden size is known). 
        """
        router, attr = router_of(self.block)
        self.route_fn = build_route_fn(router, self.top_k, self.scoring_func,
                                      self.norm_topk_prob)
        self.expert_fn, self.n_experts, self.layout = build_expert_fn(
            self.block, hidden_size
        )
        self.router_attr = attr

    @torch.inference_mode()
    def hook(
        self, 
        module: nn.Module, 
        args: tuple, 
        kwargs: dict, 
        output: torch.Tensor
    ):
        """
        Forward hook that fills per-token routing stats for one MoE block.
        The hook ignores the block's output and reconstructs the routed part
        from the block's input and the router/expert adapters.

        Block input is flattened to [T, H] where T = batch_size * seq_len 
        and H = hidden_size.

        Args:
            module: The MoE block module.
            args: Positional arguments passed to the block's forward.
            kwargs: Keyword arguments passed to the block's forward.
            output: The block's output (ignored).
        """
        h = args[0] if args else kwargs.get("hidden_states")
        if h is None:
            raise RuntimeError("MoE block called without positional or "
                               "`hidden_states` input; cannot probe it")
        H = h.shape[-1]
        h = h.reshape(-1, H)          # model dtype (bf16), on device
        T = h.shape[0]

        if self.route_fn is None:
            self._build(H)

        # Route the input through the router.
        top_k_idxs, top_exp_weights, full_activation_scores = self.route_fn(h)
        top_k = top_k_idxs.shape[-1]

        # Neutralize out-of-bounds expert indices 
        out_of_bounds = top_k_idxs >= self.n_experts
        if out_of_bounds.any():
            top_k_idxs = top_k_idxs.masked_fill(out_of_bounds, 0)
            top_exp_weights = top_exp_weights.masked_fill(out_of_bounds, 0.0)

        # Diagnostics: the full [T, E] router distribution
        if self.diagnostics:
            if full_activation_scores is None: # router only returned top-k weights
                # Scatter the top-k weights into a zeroed [T, E] tensor.
                full_activation_scores = torch.zeros(
                    T, self.n_experts, dtype=torch.float32, device=h.device
                ).scatter_(1, top_k_idxs, top_exp_weights)
                self.scores_scattered = True
            else:
                self.scores_scattered = False
            self.scores = full_activation_scores.to(torch.float16).cpu()  # this forward's chunk

        # Compute the norms of the routed activations and the delta they produce.
        norms = torch.zeros(T, top_k, dtype=torch.float32, device=h.device)
        routed_delta = torch.zeros(T, H, dtype=torch.float32, device=h.device)
        for i in range(top_k):
            top_ith_experts = top_k_idxs[:, i] # top i-th expert for each token
            for expert in top_ith_experts.unique().tolist(): 
                # For each expert, select the tokens routed to it and call the expert_fn.
                token_idxs = (top_ith_experts == expert).nonzero(as_tuple=True)[0]
                expert_output = self.expert_fn(expert, h[token_idxs]).float()      # [n, H]
                norms[token_idxs, i] = expert_output.norm(dim=-1)
                # Cumulative routed delta: add the expert's output weighted by the token's top-k weight.
                routed_delta[token_idxs] += expert_output * top_exp_weights[token_idxs, i, None]

        if self.verify and self.verdict is None:
            self.verdict = (
                "skipped (per-expert modules called directly)"
                if self.layout == "module-list"
                else verify_fused(module.experts, h, top_k_idxs,
                                  top_exp_weights, routed_delta))

        # Token contribution: 1 - cos(h, h + routed_delta) downstream; ~0 where
        # the routed experts barely move the token. Recorded here as the sim.
        h_f = h.float()
        sim = F.cosine_similarity(h_f, h_f + routed_delta, dim=-1)

        self._chunks.append((top_k_idxs.cpu(), top_exp_weights.cpu(),
                             norms.cpu(), sim.cpu(),
                             self.scores if self.diagnostics else None))
        self.chunk_sizes.append(T)
