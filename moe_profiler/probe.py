"""MoEProbe: forward hooks that record per-token routing stats for one MoE block.

Row t of a finished generation is the token at position t of prompt + answer. The
hooks never make the CPU wait for the GPU, so they can be recorded into a CUDA graph.
"""
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .discovery import router_of
from .experts import build_expert_fn, build_grouped_fn, verify_fused
from .routing import build_route_fn
from .store import pack_selected

#: (token, expert) pairs whose expert outputs are materialized at once. Bounds
#: the probe's transient memory on a long prefill (~0.5 GB at H=2048); a decode
#: step is a handful of pairs and always one piece.
PAIRS_PER_PIECE = 16384


def _rows(out) -> int | None:
    """Rows of a router output whose tensors are all [T, *], else None."""
    tensors = [out] if torch.is_tensor(out) else [
        t for t in (out if isinstance(out, (tuple, list)) else ()) if torch.is_tensor(t)]
    if not tensors or any(t.ndim != 2 for t in tensors):
        return None
    rows = {t.shape[0] for t in tensors}
    return rows.pop() if len(rows) == 1 else None


class MoEProbe:
    """
    One instance per MoE block; forward hooks fill per-token routing stats.

    The probe is initialized with the MoE block and the parameters for the routing
    and expert functions. `attach` then hooks it onto the block.
    """

    def __init__(self,
        block: nn.Module,
        top_k: int,
        scoring_func: str,
        norm_topk_prob: bool,
        verify: bool = True,
        top_m: int | None = None,
    ):
        """
        Initialize a probe for one MoE block.

        Args:
            block: The MoE block module to probe.
            top_k: The number of top experts to consider for routing.
            scoring_func: The scoring function to use for routing.
            norm_topk_prob: Whether to normalize the top-k probabilities.
            verify: Whether to verify the fused reconstruction against experts' forward.
            top_m: Router-distribution columns kept per token, descending by
                logit. Held to at least the router's k (the whole selection)
                and at most the expert count. None keeps just the selection.
        """
        self.block = block
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.verify = verify
        self.top_m = top_m
        self.route_fn = None
        self.expert_fn = None
        self.grouped_fn = None
        self.n_experts = None
        self.n_selected = None  # experts per token, as the router selected them
        self.layout = None
        self.router_attr = None
        self.verdict = None
        self.scores_scattered = None  # True if the stored distribution is a top-k
                                      # scatter, not the router's own logits
        self.host_seconds = 0.0       # host time spent in the hooks, all calls
        self._router_out = None
        self._columns = None          # M, the stored distribution columns
        self._ring = None             # device buffers for graph-replayed steps
        self._ring_pos = None         # [1] long: the next ring row, kept on device
        self.reset()

    def reset(self):
        """Reset the probe's recorded stats. Call this before each demonstration."""
        self.top_idx = None   # [T, M] long   -- experts, descending by router logit
        self.top_val = None   # [T, M] float  -- the router's values at them
        self.norms = None     # [T, k] float
        self.sim = None       # [T] float
        self._chunks = []     # one entry per eager forward, fields as in `finalize`
        self.chunk_sizes = [] # tokens seen per eager forward; [0] is the prefill length
        self._router_out = None
        self._ring_start = None  # where the ring's rows fall among the chunks
        self._ring_open = False  # steps go to the ring, replayed or not
        if self._ring_pos is not None:
            self._ring_pos.zero_()

    def use_ring(self, capacity: int) -> None:
        """
        Give the probe ``capacity`` rows of device buffers for decode steps
        replayed from a CUDA graph. Call it after the probe has seen a forward
        (the row widths come from it) and before the graph is recorded; the
        recorded hook writes into these buffers on every replay.
        """
        if self._ring is not None and self._ring[0].shape[0] == capacity:
            return  # a new recording in the same generation keeps its rows
        if self._columns is None or not self._chunks:
            raise RuntimeError("the probe needs one eager forward before a ring")
        device = self._chunks[-1][0].device
        self._ring = (
            torch.empty(capacity, self._columns, dtype=torch.long, device=device),
            torch.empty(capacity, self._columns, dtype=torch.float32, device=device),
            torch.empty(capacity, self.n_selected, dtype=torch.float32, device=device),
            torch.empty(capacity, dtype=torch.float32, device=device),
        )
        self._ring_pos = torch.zeros(1, dtype=torch.long, device=device)

    def drop_ring(self) -> None:
        """Free the ring, once no graph writes into it."""
        self._ring = self._ring_pos = None
        self._ring_start = None
        self._ring_open = False

    def mark_ring(self) -> None:
        """The graph is about to replay its first step of this generation: the
        ring's rows go after every chunk recorded so far, and every step from
        here on goes to the ring."""
        if self._ring_start is None:
            self._ring_start = len(self._chunks)
            self._ring_open = True

    def close_ring(self) -> None:
        """The ring is full: steps from here on go back to chunks, after it."""
        self._ring_open = False

    def attach(self) -> list:
        """
        Hook the probe onto its block and return the hook handles.

        Besides the block's own hook, the router is hooked to keep its output,
        so the block hook reads the routing the model actually used rather than
        running the router a second time.
        """
        handles = [self.block.register_forward_hook(self.hook, with_kwargs=True)]
        router, _ = router_of(self.block)
        if router is not None:
            handles.append(router.register_forward_hook(self._keep_router_output))
        return handles

    def _keep_router_output(self, module, args, output):
        self._router_out = output

    def finalize(self):
        """
        Concatenate the per-forward chunks into the [T, ...] per-demonstration
        stats. Call once after `generate` returns, before reading the stats.
        Leaves everything None when the block never fired.
        """
        chunks = self._chunks
        if self._ring_start is not None:
            n = int(self._ring_pos.item())
            chunks = (chunks[:self._ring_start] + [tuple(b[:n] for b in self._ring)]
                      + chunks[self._ring_start:])
        if not chunks:
            return
        idx, val, norms, sim = zip(*chunks)
        self.top_idx, self.top_val, self.norms, self.sim = (
            torch.cat(t) for t in (idx, val, norms, sim))
        self._chunks = []

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
        self.grouped_fn = build_grouped_fn(self.block, hidden_size)
        self.router_attr = attr

    def _expert_outputs(self, h: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        """``[T, k, H]``: each token's selected experts' outputs, unweighted."""
        if self.grouped_fn is not None:
            return self.grouped_fn(h, idxs)
        # No grouped GEMM for this layout or device: one expert at a time, but
        # sorted by expert first, so the only wait is for the group sizes.
        T, k = idxs.shape
        ids, order = torch.sort(idxs.reshape(-1))
        x = h.index_select(0, order // k)
        counts = torch.bincount(ids, minlength=self.n_experts).tolist()
        outs, start = [], 0
        for e, n in enumerate(counts):
            if n:
                outs.append(self.expert_fn(e, x[start:start + n]))
                start += n
        out = torch.cat(outs)
        return torch.empty_like(out).index_copy_(0, order, out).view(T, k, -1)

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
        started = time.perf_counter()
        h = args[0] if args else kwargs.get("hidden_states")
        if h is None:
            raise RuntimeError("MoE block called without positional or "
                               "`hidden_states` input; cannot probe it")
        H = h.shape[-1]
        h = h.reshape(-1, H)          # model dtype (bf16), on device
        T = h.shape[0]

        if self.route_fn is None:
            self._build(H)

        # The router's output from this very forward when the block ran it on
        # these tokens, else the router run again on the block's input.
        router_out, self._router_out = self._router_out, None
        top_k_idxs, top_exp_weights, full_activation_scores = self.route_fn(
            h, router_out if router_out is not None and _rows(router_out) == T else None)
        k = top_k_idxs.shape[-1]
        E = self.n_experts

        # The stored distribution: its top-m columns, descending.
        if full_activation_scores is not None and full_activation_scores.shape[-1] == E:
            # The router's own selection leads, in its order, then the best of
            # the rest. Ranking the logits afresh would agree but for ties --
            # common among bf16 logits -- which it could break the other way,
            # storing an expert the token never went to.
            self.scores_scattered = False
            m = min(max(self.top_m or k, k), E)
            top_idx = top_k_idxs
            top_val = full_activation_scores.gather(1, top_k_idxs)
            if m > k:
                rest_val, rest_idx = torch.topk(
                    full_activation_scores.scatter(1, top_k_idxs, float("-inf")),
                    m - k, dim=-1)
                top_idx = torch.cat([top_idx, rest_idx], dim=-1)
                top_val = torch.cat([top_val, rest_val], dim=-1)
        else:
            # Neutralize out-of-bounds expert indices: sentinels some routers
            # use for a dropped slot. A router that scored exactly the E experts
            # it chose from cannot produce one.
            out_of_bounds = top_k_idxs >= E
            top_k_idxs = top_k_idxs.masked_fill(out_of_bounds, 0)
            top_exp_weights = top_exp_weights.masked_fill(out_of_bounds, 0.0)
            if full_activation_scores is None:  # router only returned top-k weights
                # Scatter the top-k weights into a zeroed [T, E] tensor.
                full_activation_scores = torch.zeros(
                    T, E, dtype=torch.float32, device=h.device
                ).scatter_(1, top_k_idxs, top_exp_weights)
                self.scores_scattered = True
            else:
                self.scores_scattered = False
            m = min(max(self.top_m or k, k), full_activation_scores.shape[-1])
            top_val, top_idx = torch.topk(full_activation_scores, m, dim=-1)

        # The norms of the routed experts' outputs, and the delta they add to
        # the token, in pieces that bound the transient memory.
        per_piece = max(1, PAIRS_PER_PIECE // k)
        norms, deltas = [], []
        for s in range(0, T, per_piece):
            out = self._expert_outputs(h[s:s + per_piece],
                                       top_k_idxs[s:s + per_piece]).float()  # [t, k, H]
            norms.append(torch.linalg.vector_norm(out, dim=-1))
            deltas.append((out * top_exp_weights[s:s + per_piece, :, None]).sum(1))
        norms = norms[0] if len(norms) == 1 else torch.cat(norms)
        routed_delta = deltas[0] if len(deltas) == 1 else torch.cat(deltas)

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

        self.n_selected, self._columns = k, top_idx.shape[-1]
        if self._ring_open or torch.cuda.is_current_stream_capturing():
            # Being recorded into a CUDA graph -- this runs once, the graph many
            # times -- or a step between replays: the rows go to the ring, at
            # the position the device keeps.
            if self._ring is None:
                raise RuntimeError("probe recorded into a CUDA graph without a "
                                   "ring to write into; call use_ring first")
            rows = self._ring_pos + torch.arange(T, device=h.device)
            for buf, x in zip(self._ring, (top_idx, top_val, norms, sim)):
                buf.index_copy_(0, rows, x)
            self._ring_pos.add_(T)
        else:
            self._chunks.append((top_idx, top_val, norms, sim))
            self.chunk_sizes.append(T)
        self.host_seconds += time.perf_counter() - started


def pack_probes(probes: list[MoEProbe]) -> dict[str, np.ndarray]:
    """
    One generation's stats, stacked over the probes' layers and packed for
    `store.write_generation`. The probes must be finalized. This is where the
    host waits for the device: once per generation, not once per token.
    """
    def stacked(name: str) -> np.ndarray:
        # Under a device_map split the blocks, and so their stats, live on
        # different GPUs: gather them on the first one, then copy over once.
        device = getattr(probes[0], name).device
        return torch.stack([getattr(p, name).to(device) for p in probes]).cpu().numpy()

    return pack_selected(
        stacked("top_idx"),  # [L, T, M]
        stacked("top_val"),  # [L, T, M]
        stacked("norms"),    # [L, T, k]
        stacked("sim"),      # [L, T]
        n_experts=probes[0].n_experts,
        scattered=bool(probes[0].scores_scattered),
    )
