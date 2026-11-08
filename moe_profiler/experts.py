"""Compute selected experts' outputs, whatever layout a MoE block stores them in.

`build_expert_fn` computes one expert at a time; `build_grouped_fn` computes every
selected expert of every token at once.
"""
import inspect
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def _orientation(weight, in_features):
    """Is this [E, out, in] (F.linear) or [E, in, out] (matmul)? -> transposed?"""
    _, a, b = weight.shape
    if b == in_features and a != in_features:
        return False
    if a == in_features and b != in_features:
        return True
    if a == b == in_features:   # square: both readings work, F.linear is the norm
        return False
    raise RuntimeError(f"expert weight {tuple(weight.shape)} matches neither "
                       f"[E,out,{in_features}] nor [E,{in_features},out]")


def _project(x, w, b, transposed):
    if transposed:
        y = x @ w
        return y if b is None else y + b
    return F.linear(x, w, b)


def _act_from_config(block):
    try:
        from transformers.activations import ACT2FN
    except ImportError:
        return F.silu
    cfg = getattr(block, "config", None)
    name = getattr(cfg, "hidden_act", None) or getattr(cfg, "hidden_activation", None)
    return ACT2FN[name] if name in getattr(ACT2FN, "keys", lambda: ())() else F.silu



def _fused_spec(block: nn.Module, hidden_size: int) -> dict | None:
    """
    Read a fused expert layout off ``block.experts``: its stacked weights and
    biases, their orientation, and the activation. None for a per-expert
    ModuleList, which has no stacked weights to read.
    """
    experts = block.experts
    if isinstance(experts, (nn.ModuleList, nn.Sequential)):
        if len(experts) and isinstance(experts[0], nn.Module):
            return None

    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    gate_only = getattr(experts, "gate_proj", None)
    up_only = getattr(experts, "up_proj", None)
    if down is None or (gate_up is None and (gate_only is None or up_only is None)):
        raise RuntimeError(
            f"unsupported expert layout on {type(experts).__name__}: expected an "
            f"nn.ModuleList, or fused `gate_up_proj`+`down_proj`, or fused "
            f"`gate_proj`+`up_proj`+`down_proj`. Attributes: "
            f"{[n for n, _ in experts.named_parameters(recurse=False)]}")

    for tag, t in (("gate_up_proj", gate_up), ("down_proj", down),
                   ("gate_proj", gate_only), ("up_proj", up_only)):
        if t is not None and t.device.type == "meta":
            raise RuntimeError(
                f"{tag} is a meta tensor -- the checkpoint is offloaded and the "
                f"fused weights cannot be read directly. Load without CPU/disk "
                f"offload (raise max_memory) for the stats pass.")

    fused_gate_up = gate_up is not None
    ref = gate_up if fused_gate_up else gate_only
    gu_t = _orientation(ref, hidden_size)
    # The out-features axis is the one that is NOT hidden_size.
    inter = (ref.shape[2] if gu_t else ref.shape[1])
    if fused_gate_up:
        inter //= 2
    spec = {
        "gate_up": gate_up, "gate": gate_only, "up": up_only, "down": down,
        "gu_bias": getattr(experts, "gate_up_proj_bias", None),
        "dn_bias": getattr(experts, "down_proj_bias", None),
        "g_bias": getattr(experts, "gate_proj_bias", None),
        "u_bias": getattr(experts, "up_proj_bias", None),
        "gu_t": gu_t,
        "dn_t": _orientation(down, inter),
        "act": getattr(experts, "act_fn", None) or _act_from_config(block),
        "n_experts": int(getattr(experts, "num_experts", None) or ref.shape[0]),
    }
    spec["layout"] = (f"fused-contiguous"
                      f"{'' if fused_gate_up else '-split'}"
                      f"{'-transposed' if gu_t else ''}"
                      f"{'-biased' if spec['gu_bias'] is not None or spec['dn_bias'] is not None else ''}")
    return spec


def build_expert_fn(block: nn.Module, hidden_size: int):
    """
    Build ``expert_fn(e, hs) -> [n, H]`` and report (fn, n_experts, layout).
    """
    spec = _fused_spec(block, hidden_size)

    # --- per-expert ModuleList: call the real module, exact by construction ---
    if spec is None:
        experts = block.experts
        return (lambda e, hs: experts[e](hs)), len(experts), "module-list"

    # --- fused tensors: reconstruct, detecting every layout choice ------------
    gate_up, gate_only, up_only, down = (spec[k] for k in ("gate_up", "gate", "up", "down"))
    gu_bias, dn_bias, g_bias, u_bias = (spec[k] for k in ("gu_bias", "dn_bias", "g_bias", "u_bias"))
    gu_t, dn_t, act = spec["gu_t"], spec["dn_t"], spec["act"]
    fused_gate_up = gate_up is not None

    def expert_fn(e, hs):
        if fused_gate_up:
            y = _project(hs, gate_up[e], None if gu_bias is None else gu_bias[e], gu_t)
            g_, u_ = y.chunk(2, dim=-1)
        else:
            g_ = _project(hs, gate_only[e], None if g_bias is None else g_bias[e], gu_t)
            u_ = _project(hs, up_only[e], None if u_bias is None else u_bias[e], gu_t)
        return _project(act(g_) * u_, down[e], None if dn_bias is None else dn_bias[e], dn_t)

    return expert_fn, spec["n_experts"], spec["layout"]


#: Grouped GEMMs of up to this many rows -- a decode step has top-k per MoE block
#: -- gather their experts' weights and run one batched matmul instead, on a
#: device whose own grouped GEMM cannot be recorded into a CUDA graph.
GATHER_ROWS = 64

_capturable: dict = {}


def grouped_mm_capturable(device: torch.device) -> bool:
    """
    Can torch's grouped GEMM on this device be recorded into a CUDA graph?

    Where torch has no native kernel for the GPU (SM 120 and SM 75 in torch 2.13,
    for two) it runs a fallback that reads the group offsets back to the host,
    which a recording cannot contain. Found out once per device, by trying; call
    it outside any recording, since the answer is only cached afterwards.
    """
    if device.type != "cuda" or not hasattr(F, "grouped_mm"):
        return True
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _capturable:
        with torch.cuda.device(index):
            w = torch.zeros(2, 16, 16, dtype=torch.bfloat16, device="cuda")
            x = torch.zeros(4, 16, dtype=torch.bfloat16, device="cuda")
            offs = torch.tensor([2, 4], dtype=torch.int32, device="cuda")
            try:
                F.grouped_mm(x, w, offs=offs)
                # A failed recording ends as an empty graph, which torch warns of.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with torch.cuda.graph(torch.cuda.CUDAGraph(),
                                          capture_error_mode="thread_local"):
                        F.grouped_mm(x, w, offs=offs)
                _capturable[index] = True
            except Exception:  # noqa: BLE001 -- any failure means: not recordable
                _capturable[index] = False
    return _capturable[index]


def gathered_mm(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """
    ``F.grouped_mm(x, w, offs=offs)`` for a few rows, with the host never waiting
    on the device: each row's expert weights are gathered and one batched matmul
    runs over them. Costs a copy of the gathered weights, so only for few rows.
    """
    rows = torch.arange(x.shape[0], device=x.device, dtype=offs.dtype)
    e = torch.searchsorted(offs, rows, right=True).clamp_max_(w.shape[0] - 1)
    # Gather along the layout the weights are stored in: a transposed view is
    # gathered from its contiguous base and transposed back, which bmm takes.
    wt = w.transpose(-2, -1)
    we = wt.index_select(0, e).transpose(-2, -1) if wt.is_contiguous() else w.index_select(0, e)
    return torch.bmm(x.unsqueeze(1), we).squeeze(1)


def grouped_mm(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """``F.grouped_mm``, but recordable into a CUDA graph for a few rows on a
    device where torch's own is not (see `grouped_mm_capturable`)."""
    if x.is_cuda and x.shape[0] <= GATHER_ROWS and not _capturable.get(
            x.device.index if x.device.index is not None else torch.cuda.current_device(), True):
        return gathered_mm(x, w, offs)
    return F.grouped_mm(x, w, offs=offs)


def _grouped_mm_runs(weight: torch.Tensor, transposed: bool) -> bool:
    """Can this device run a grouped GEMM against these stacked weights?"""
    if not hasattr(F, "grouped_mm"):
        return False
    n_in = weight.shape[1] if transposed else weight.shape[2]
    try:
        F.grouped_mm(weight.new_zeros(1, n_in),
                     weight if transposed else weight.transpose(-2, -1),
                     offs=torch.ones(weight.shape[0], dtype=torch.int32,
                                     device=weight.device))
        return True
    except (RuntimeError, NotImplementedError, TypeError):
        return False


def build_grouped_fn(block: nn.Module, hidden_size: int):
    """
    Build ``grouped_fn(h, idxs) -> [T, k, H]``: the output of every selected
    expert for every token, unweighted, with the host never waiting on the device.

    The (token, slot) pairs are sorted by expert, so each expert's tokens are one
    contiguous run, and each projection is then a single grouped GEMM over all of
    them -- the same computation `expert_fn` does one expert at a time. Returns
    None for a per-expert ModuleList, or when the device cannot run grouped GEMMs
    on these weights; the caller then goes expert by expert.
    """
    spec = _fused_spec(block, hidden_size)
    if spec is None:
        return None
    gate_up, gate_only, up_only, down = (spec[k] for k in ("gate_up", "gate", "up", "down"))
    gu_bias, dn_bias, g_bias, u_bias = (spec[k] for k in ("gu_bias", "dn_bias", "g_bias", "u_bias"))
    gu_t, dn_t, act = spec["gu_t"], spec["dn_t"], spec["act"]
    fused_gate_up = gate_up is not None
    if not all(_grouped_mm_runs(w, t) for w, t in (
            (gate_up if fused_gate_up else gate_only, gu_t), (down, dn_t))):
        return None
    grouped_mm_capturable(down.device)      # settled now, outside any recording
    # Built once: the searchsorted keys that turn sorted expert ids into offsets.
    experts = torch.arange(spec["n_experts"], device=down.device)

    def project(x, w, b, ids, offs, transposed):
        y = grouped_mm(x, w if transposed else w.transpose(-2, -1), offs)
        return y if b is None else y + b[ids]

    def grouped_fn(h, idxs):
        T, k = idxs.shape
        ids, order = torch.sort(idxs.reshape(-1))
        x = h.index_select(0, order // k)                  # [T*k, H], by expert
        # offs[e]: pairs routed to experts 0..e, i.e. where expert e's run ends.
        offs = torch.searchsorted(ids, experts, right=True).to(torch.int32)
        if fused_gate_up:
            g_, u_ = project(x, gate_up, gu_bias, ids, offs, gu_t).chunk(2, dim=-1)
        else:
            g_ = project(x, gate_only, g_bias, ids, offs, gu_t)
            u_ = project(x, up_only, u_bias, ids, offs, gu_t)
        out = project(act(g_) * u_, down, dn_bias, ids, offs, dn_t)
        # Back to (token, slot) order; `order` is a permutation, so no two rows
        # land on the same place.
        return torch.empty_like(out).index_copy_(0, order, out).view(T, k, -1)

    return grouped_fn



def verify_fused(
    experts: torch.nn.Module, 
    h: torch.Tensor, 
    idxs: torch.Tensor, 
    weights: torch.Tensor, 
    recon: torch.Tensor
) -> str:
    """
    Best-effort cross-check of a fused reconstruction against experts' forward.

    Returns a human-readable verdict. Fused expert modules do not share a
    signature across model families, so a failure here means "could not
    compare", not "reconstruction is wrong".
    """
    try:
        params = list(inspect.signature(experts.forward).parameters)
        if len(params) < 3:
            return "skipped (experts.forward takes no routing args)"
        ref = experts(h, idxs, weights.to(h.dtype))
        ref = ref[0] if isinstance(ref, tuple) else ref
        ref = ref.reshape(recon.shape).float()
        denom = ref.norm().clamp_min(1e-6)
        err = (ref - recon).norm() / denom
        return (f"OK (rel err {err:.2e})" if err < 5e-2
                else f"MISMATCH (rel err {err:.2e}) -- check the expert layout, "
                     f"or ignore if experts.forward uses a different weight convention")
    except Exception as exc:
        return f"skipped ({type(exc).__name__}: {exc})"
