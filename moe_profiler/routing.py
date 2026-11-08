"""Read any router's output as top-k expert ids, their weights, and the full distribution.
"""
import torch
from torch import nn

INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)

def _weights_from_logits(
    logits: torch.Tensor, 
    top_k: int, 
    scoring_func: str, 
    norm_topk_prob: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize a router's logits to its weights.
    
    Args:
        logits: The raw logits output from the router, shape [T, E].
        top_k: The number of top experts to select.
        scoring_func: The scoring function to use ('softmax' or 'sigmoid').
        norm_topk_prob: Whether to normalize the top-k probabilities.
    
    Returns:
        A tuple of (idxs, weights, scores_full):
            idxs: The indices of the top-k experts, shape [T, k].
            weights: The normalized weights for the top-k experts, shape [T, k].
            scores_full: The full distribution over all experts, shape [T, E].
    """
    logits = logits.float()
    scores = logits.sigmoid() if scoring_func == "sigmoid" else logits.softmax(dim=-1)
    weights, idxs = torch.topk(scores, top_k, dim=-1)
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return idxs, weights, scores



def _split_router_output(
    out: torch.Tensor, 
    top_k: int, 
    scoring_func: str,
    norm_topk_prob: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Extract a router's top-k indices, corresponding weights, and optionally
    its full activation distribution from its output.

    Args:
        out: The raw output from the router, which can be a tensor or a tuple/list
            containing tensors.
        top_k: The number of top experts to select.
        scoring_func: The scoring function to use ('softmax' or 'sigmoid').
        norm_topk_prob: Whether to normalize the top-k probabilities.
    
    Returns:
        A tuple of (idxs, weights, scores_full):
            idxs: The indices of the top-k experts, shape [T, k]. (T=sequence length, k=top_k)
            weights: The normalized weights for the top-k experts, shape [T, k].
            scores_full: The full distribution over all experts, shape [T, E], or None
    """
    # If the router returned a single tensor, it must be the full [T,E] logits.
    if torch.is_tensor(out):
        return _weights_from_logits(out, top_k, scoring_func, norm_topk_prob)

    # If the router returned a tuple or list, it must contain at least the top-k
    # indices and weights, and optionally the full [T,E] distribution. 
    if not isinstance(out, (tuple, list)):
        out = tuple(getattr(out, f, None) for f in
                    ("router_indices", "router_scores", "routing_weights", "logits"))

    tensors = [t for t in out if torch.is_tensor(t)]
    # The top-k indices are the only integer tensor of shape [T, k].
    idxs = next((t for t in tensors if t.dtype in INT_DTYPES and t.ndim == 2), None)

    # No indices returned -> the only tensor is a logit/score matrix.
    if idxs is None:
        logits = next((t for t in tensors if t.ndim == 2), None)
        if logits is None:
            raise RuntimeError(f"cannot interpret router output of type {type(out)}")
        return _weights_from_logits(logits, top_k, scoring_func, norm_topk_prob)

    k = idxs.shape[-1]
    # The full activations distribution, if present, is the widest float matrix ([T, E>k]).
    scores_full = next((t for t in tensors
                        if t.dtype not in INT_DTYPES and t.ndim == 2
                        and t.shape[0] == idxs.shape[0] and t.shape[-1] > k), None)
    weights = next((t for t in tensors
                    if t.dtype not in INT_DTYPES and t.shape == idxs.shape), None)
    if weights is None:
        # Only full [T, E] scores plus [T, k] indices -> gather the selected weights.
        if scores_full is None:
            raise RuntimeError("router returned indices but no usable weights")
        weights = scores_full.gather(-1, idxs)
    return (idxs.long(), weights.float(),
            scores_full.float() if scores_full is not None else None)



def build_route_fn(
    router: nn.Module, 
    top_k: int, 
    scoring_func: str, 
    norm_topk_prob: bool
) -> callable:
    """
    Build a route function that extracts top-k indices, weights, and 
    optionally the full distribution from a router's output.

    Args:
        router: The router module that produces logits or scores.
        top_k: The number of top experts to select.
        scoring_func: The scoring function to use ('softmax' or 'sigmoid').
        norm_topk_prob: Whether to normalize the top-k probabilities.

    Returns:
        A function that takes input tensor h and returns (idxs, weights,
        scores_full). Passing it the router's own output on h as ``out`` skips
        running the router again.
    """
    def route(h, out=None):
        return _split_router_output(router(h) if out is None else out,
                                    top_k, scoring_func, norm_topk_prob)
    return route
