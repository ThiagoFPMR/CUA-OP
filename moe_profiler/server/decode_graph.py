"""Faster decoding: replay each decode step from a CUDA graph.

Launching a step's thousands of small kernels from Python is slower than running
them; a CUDA graph replays them in one launch. `DecodeGraph` plugs into
transformers' compiled-decoding hook, and falls back to normal decoding when a
step cannot be recorded.
"""
from __future__ import annotations

import contextlib
import logging

import torch
from torch import nn
from torch.utils import _pytree as pytree
from transformers import StaticCache
from transformers.cache_utils import LinearAttentionLayer, StaticLayer

from ..experts import GATHER_ROWS, gathered_mm, grouped_mm_capturable

logger = logging.getLogger("moe_profiler.server")


def _route_expert_gemms(device: torch.device):
    """Point transformers' grouped expert GEMM at one a CUDA graph can record,
    if the device's own cannot be. Returns the function to restore, or None."""
    try:
        from transformers.integrations import moe
    except ImportError:
        return None
    if grouped_mm_capturable(device) or not hasattr(moe, "_grouped_mm"):
        return None
    original = moe._grouped_mm

    def _grouped_mm(input, weight, offs):
        if input.shape[0] <= GATHER_ROWS:
            return gathered_mm(input.to(weight.dtype), weight, offs)
        return original(input, weight, offs=offs)

    moe._grouped_mm = _grouped_mm
    logger.info("decode graphs: expert GEMMs of decode steps use gathered weights "
                "(torch's own grouped GEMM cannot be recorded on this GPU)")
    return original


# The roles of a decoder layer's inputs within a segment's recording.
HIDDEN = "hidden"   # the previous layer's output: carried inside the graph
CONST = "const"     # not a tensor (the cache, flags): fixed by the recording


def _hidden_states(args: tuple, kwargs: dict):
    return args[0] if args else kwargs.get("hidden_states")


def _signature(leaf):
    """What a recording assumes about one non-hidden input of a layer."""
    if torch.is_tensor(leaf):
        return (leaf.shape, leaf.dtype, leaf.device)
    if leaf is None or isinstance(leaf, (bool, int, float, str)):
        return (type(leaf), leaf)
    return id(leaf)


def _decoder_layers(model: nn.Module, blocks: list) -> nn.ModuleList | None:
    """The module list holding the decoder layers: the one whose children
    hold the most of the probed MoE blocks (a vision tower's hold none)."""
    wanted = {id(b) for b in blocks}
    best, hits = None, 0
    for module in model.modules():
        if isinstance(module, nn.ModuleList):
            n = sum(any(id(m) in wanted for m in child.modules()) for child in module)
            if n > hits:
                best, hits = module, n
    return best


def _run_chain(forwards: list, flat: list, placeholders: list):
    """Run a segment's layers one after another on ``flat`` -- each layer's
    inputs as (pytree leaves, spec) -- feeding each layer the previous one's
    output where the model's loop handed it the previous placeholder."""
    h = None
    for p, (forward, (leaves, spec)) in enumerate(zip(forwards, flat)):
        if p:
            leaves = [h if x is placeholders[p - 1] else x for x in leaves]
        args, kwargs = pytree.tree_unflatten(leaves, spec)
        h = forward(*args, **kwargs)
    return h


class _Segment:
    """A run of consecutive decoder layers on one GPU, and its recording."""

    def __init__(self, device: torch.device, forwards: list):
        self.device = device
        self.forwards = forwards  # the layers' own forwards (accelerate's hooks included)
        self.drop()

    def drop(self) -> None:
        self.graph = self.output = None
        self.slots = []       # static device copies of the layers' tensor inputs
        self.roles = []       # per layer, per input leaf: HIDDEN, CONST or a slot index
        self.signature = []   # per layer: (spec, per leaf `_signature` or HIDDEN)


class LayerSegments:
    """
    Replays a decode step's decoder layers from one CUDA graph per GPU, for a
    model spread over several.

    Each layer's forward is wrapped. Outside a decode step (the prefill, eager
    steps) the wrappers only pass through. In a recorded or replayed step, the
    model's own forward runs as usual up to the layers; each layer but a
    segment's last then just notes its inputs and hands the loop a placeholder
    for its output, and the segment's last layer runs the whole segment: copies
    the noted inputs into the segment's static copies on its GPU and replays
    its graph (or first records it). Inputs coming from another GPU -- the
    hidden state, the rotary embeddings, the mask -- are copied over eagerly,
    between graphs, so no recording spans two devices.

    The first eager step of every recording key checks, once, that the model
    runs its layers the way this relies on: in order, each layer's first input
    the previous one's output, each returning a single tensor. Anything else
    leaves the decode steps eager.
    """

    def __init__(self, layers: nn.ModuleList, segments: list[_Segment],
                 probes: list):
        self.layers = layers
        self.segments = segments
        self.probes = probes
        self.validated = False
        self._saved = None
        self._mode = None           # None | "warm" | "record" | "replay"
        self._pending: list = []    # the current segment's (args, kwargs) so far
        self._placeholders: list = []
        self.failure: BaseException | None = None
        self._warm_calls: list = []
        self._warm_prev = None
        self._warm_inputs: dict = {}  # id -> (tensor input, its contents before)
        self._problem: str | None = None

    @classmethod
    def build(cls, model: nn.Module, probes: list) -> LayerSegments | None:
        """Segments for ``model`` if its parameters span several GPUs, else None."""
        if len({p.device for p in model.parameters()}) < 2:
            return None
        layers = _decoder_layers(model, [p.block for p in probes])
        if layers is None:
            logger.warning("decode graphs: model spans several devices but its "
                           "decoder layers were not found; decoding eagerly")
            return None
        segments, forwards, device = [], [], None
        for layer in layers:
            devices = {p.device for p in layer.parameters()}
            if len(devices) != 1 or next(iter(devices)).type != "cuda":
                logger.warning("decode graphs: a decoder layer is not on exactly "
                               "one GPU (%s); decoding eagerly", devices)
                return None
            (layer_device,) = devices
            if layer_device != device:
                forwards = []
                segments.append(_Segment(layer_device, forwards))
                device = layer_device
            forwards.append(layer.forward)
        logger.info("decode graphs: %d decoder layers recorded as %d per-GPU "
                    "segments (%s)", len(layers), len(segments),
                    ", ".join(f"{s.device}: {len(s.forwards)}" for s in segments))
        return cls(layers, segments, probes)

    # --------------------------------------------------------------- hooking

    def install(self) -> None:
        if self._saved is not None:
            return
        self._saved = []
        position = 0
        for s, segment in enumerate(self.segments):
            for p, forward in enumerate(segment.forwards):
                layer = self.layers[position]
                self._saved.append((layer, layer.__dict__.get("forward")))
                layer.forward = self._wrap(s, p, forward)
                position += 1

    def uninstall(self) -> None:
        self.drop()
        for layer, own in self._saved or ():
            if own is None:
                layer.__dict__.pop("forward", None)
            else:
                layer.forward = own
        self._saved = None

    def drop(self) -> None:
        for segment in self.segments:
            segment.drop()

    def _wrap(self, s: int, position: int, forward):
        segment = self.segments[s]
        last = position == len(segment.forwards) - 1

        def wrapped(*args, **kwargs):
            if self._mode is None:
                return forward(*args, **kwargs)
            if self._mode == "warm":
                return self._warm(s, position, forward, args, kwargs)
            if len(self._pending) != position:
                raise RuntimeError("decode graphs: decoder layers ran out of order")
            self._pending.append((args, kwargs))
            if not last:
                # A new tensor object over the input's memory, told apart from
                # every other input of the next layer by identity.
                h = _hidden_states(args, kwargs)
                self._placeholders.append(h.view_as(h))
                return self._placeholders[-1]
            try:
                return self._finish(segment)
            finally:
                self._pending, self._placeholders = [], []

        return wrapped

    # ------------------------------------------------------------- one step

    def run(self, model: nn.Module, kwargs: dict, mode: str):
        """Run one decode step of ``model`` with the layers in ``mode``: "warm"
        (eager, checking the layers' calls), "record" or "replay". A segment
        that cannot record or replay runs eagerly, and so does the rest of the
        step; `failure` then holds why."""
        self._mode, self.failure = mode, None
        self._pending, self._placeholders = [], []
        if mode == "warm":
            self._warm_calls, self._warm_prev, self._problem = [], None, None
            self._warm_inputs = {}
        try:
            return model(**kwargs)
        finally:
            self._mode, self._warm_prev = None, None
            self._pending, self._placeholders = [], []

    def warm_problem(self) -> str | None:
        """After a "warm" step: why the layers cannot be segmented, or None."""
        expected = [(s, p) for s, segment in enumerate(self.segments)
                    for p in range(len(segment.forwards))]
        if self._problem is None and self._warm_calls != expected:
            self._problem = (f"the step ran {len(self._warm_calls)} decoder layer "
                             f"calls, not each of the {len(expected)} layers once, in order")
        if self._problem is None and not all(
                torch.equal(x, before) for x, before in self._warm_inputs.values()):
            self._problem = "a decoder layer changes one of its tensor inputs in place"
        self._warm_inputs = {}
        self.validated = self._problem is None
        return self._problem

    def _warm(self, s: int, position: int, forward, args: tuple, kwargs: dict):
        h = _hidden_states(args, kwargs)
        if self._problem is None:
            leaves, _ = pytree.tree_flatten((args, kwargs))
            # A replay feeds a layer static copies of its tensor inputs, so an
            # input the layer changes in place would change only the copy.
            # Inference tensors keep no version counter: compare contents.
            for x in leaves:
                if torch.is_tensor(x) and id(x) not in self._warm_inputs:
                    self._warm_inputs[id(x)] = (x, x.clone())
            if not torch.is_tensor(h):
                self._problem = "a decoder layer's first input is not a tensor"
            elif position and (h is not self._warm_prev
                               or sum(x is h for x in leaves) != 1):
                self._problem = ("a decoder layer's hidden-state input is not the "
                                 "previous layer's output, passed once")
        self._warm_calls.append((s, position))
        out = forward(*args, **kwargs)
        if self._problem is None and not torch.is_tensor(out):
            self._problem = "a decoder layer returns more than its hidden states"
        self._warm_prev = out
        return out

    def _finish(self, segment: _Segment):
        flat = [pytree.tree_flatten((args, kwargs)) for args, kwargs in self._pending]
        try:
            if self._mode == "record":
                self._record(segment, flat)
            else:
                self._load(segment, flat)
            with torch.cuda.device(segment.device):
                segment.graph.replay()
            return segment.output
        except Exception as exc:  # noqa: BLE001 -- any failure: this step goes on eagerly
            self.failure, self._mode = exc, None
            segment.drop()
            # A recording that broke off may leave a router output its block
            # never consumed; the eager run must not pick it up.
            for probe in self.probes:
                probe._router_out = None
            return _run_chain(segment.forwards, flat, self._placeholders)

    def _record(self, segment: _Segment, flat: list) -> None:
        slots, index, roles, signature = [], {}, [], []
        for p, (leaves, spec) in enumerate(flat):
            layer_roles = []
            for x in leaves:
                if p and x is self._placeholders[p - 1]:
                    layer_roles.append(HIDDEN)
                elif torch.is_tensor(x):
                    if id(x) not in index:
                        index[id(x)] = len(slots)
                        slots.append(x.to(segment.device, copy=True))
                    layer_roles.append(index[id(x)])
                else:
                    layer_roles.append(CONST)
            if p and layer_roles.count(HIDDEN) != 1:
                raise RuntimeError("decode graphs: a decoder layer's hidden-state "
                                   "input is not the previous layer's output")
            roles.append(layer_roles)
            signature.append((spec, [HIDDEN if r == HIDDEN else _signature(x)
                                     for x, r in zip(leaves, layer_roles)]))
        static = [([slots[r] if isinstance(r, int) else x
                    for x, r in zip(leaves, layer_roles)], spec)
                  for (leaves, spec), layer_roles in zip(flat, roles)]

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(segment.device):
            # What torch.cuda.graph does, less its gc.collect(): a full
            # collection per segment per recording would cost more than the
            # launches the recording saves on a short answer.
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                # Thread-local: only this thread's own calls can spoil it.
                graph.capture_begin(capture_error_mode="thread_local")
                try:
                    output = _run_chain(segment.forwards, static, self._placeholders)
                except BaseException:
                    with contextlib.suppress(Exception):
                        graph.capture_end()
                    raise
                graph.capture_end()
            torch.cuda.current_stream().wait_stream(stream)
        segment.graph, segment.output = graph, output
        segment.slots, segment.roles, segment.signature = slots, roles, signature

    def _load(self, segment: _Segment, flat: list) -> None:
        """Copy a replayed step's inputs into the segment's static copies,
        checking they are the inputs the recording was made for."""
        if segment.graph is None:
            raise RuntimeError("decode graphs: no recording for this segment")
        filled = {}
        for p, ((leaves, spec), layer_roles, (spec0, sig)) in enumerate(
                zip(flat, segment.roles, segment.signature)):
            if spec != spec0 or len(leaves) != len(sig):
                raise RuntimeError("decode graphs: a decoder layer's inputs are "
                                   "laid out differently than when recorded")
            for x, r, s in zip(leaves, layer_roles, sig):
                if r == HIDDEN:
                    if x is not self._placeholders[p - 1]:
                        raise RuntimeError("decode graphs: a decoder layer's "
                                           "hidden-state input moved")
                elif _signature(x) != s:
                    raise RuntimeError("decode graphs: a decoder layer's inputs "
                                       "differ from the recorded ones")
                elif isinstance(r, int):
                    seen = filled.get(r)
                    if seen is None:
                        segment.slots[r].copy_(x)
                        filled[r] = x
                    elif seen is not x:
                        raise RuntimeError("decode graphs: two decoder-layer inputs "
                                           "shared when recorded differ now")


class DecodeGraph:
    """Stands in for ``model.get_compiled_call``'s result during `generate`."""

    #: What `generate` passes a decode step that the step does not read: with the
    #: positions already given, these only feed the prefill's position
    #: computation, and one grows by a token every step -- which would otherwise
    #: call for a new recording every step.
    UNUSED = ("mm_token_type_ids", "image_grid_thw", "video_grid_thw")

    #: Decode steps a generation can send to the probes' rings; past it the rest
    #: run eagerly. Sized for the whole answer budget agents ask for, so even an
    #: answer that runs away to the limit stays replayed: each ring holds this
    #: many rows, about 4 MB per MoE block (~170 MB for 40 blocks).
    RING_ROWS = 32768

    #: Answer tokens the cache has room for before it first grows. Live answers
    #: run ~250 tokens, rarely past 1000.
    ANSWER_ROOM = 1024

    def __init__(self, model, probes: list, replay: bool = True):
        """
        Args:
            model: The model whose decode steps to replay. Installs itself on it.
            probes: The probes hooked on it, which the recording must include.
            replay: False keeps the static cache but runs every step eagerly --
                the computation a replay must reproduce, to check it against.
        """
        self.model = model
        self.probes = probes
        self.enabled = replay
        self._graph = None
        self._key = None          # the inputs the recorded graph was made for
        self._warm_key = None     # the inputs the last eager warm-up step ran on
        self._inputs = {}         # static copies the graph reads its inputs from
        self._masks = {}
        self._output = None
        self._cache = None
        self._prompt = self._position = self._limit = 0
        self.replays = self.eager_steps = self.growths = 0
        self.recorded = False
        self._ring_rows = 0
        self._marked = False      # this generation's probes write to their rings
        self._original_gemm = None
        # Several GPUs: the decoder layers are recorded per GPU instead.
        self._segments = LayerSegments.build(model, probes) if replay else None
        self.install()

    def install(self) -> None:
        """Take over ``get_compiled_call`` on the model's instance, and the
        expert GEMMs if they need it."""
        self.model.get_compiled_call = lambda compile_config=None: self
        if self._original_gemm is None:
            self._original_gemm = _route_expert_gemms(next(self.model.parameters()).device)
        if self._segments is not None:
            self._segments.install()

    def uninstall(self) -> None:
        self.release()
        self.model.__dict__.pop("get_compiled_call", None)
        if self._segments is not None:
            self._segments.uninstall()
        if self._original_gemm is not None:
            from transformers.integrations import moe
            moe._grouped_mm, self._original_gemm = self._original_gemm, None

    def start(self, n_prompt: int, max_new_tokens: int) -> StaticCache:
        """
        Start a generation: returns the static cache to hand `generate` as
        ``past_key_values``, which is what routes its decode steps here.
        """
        self.replays = self.eager_steps = self.growths = self._ring_rows = 0
        self.recorded = self._marked = False
        self._prompt = self._position = n_prompt  # where the next step writes
        # The last token is never fed back, so no step writes at the limit.
        self._limit = n_prompt + max_new_tokens
        config = self.model.config.get_text_config(decoder=True)
        self._cache = StaticCache(
            config=config, max_cache_len=min(self._limit, n_prompt + self.ANSWER_ROOM))
        # Growing is only defined for plain full-attention layers; anything
        # else (a sliding window, say) gets the whole budget up front.
        if not all(type(layer) in (StaticLayer, LinearAttentionLayer)
                   for layer in self._cache.layers):
            self._cache = StaticCache(config=config, max_cache_len=self._limit)
        return self._cache

    def end(self) -> dict:
        """Close a generation: free its recording and cache before the next
        prefill needs the memory. The probes keep their rings, which hold rows
        they have yet to finalize. Returns how the steps ran, for its timings."""
        self._drop_graph()
        self._warm_key = None
        self._cache = None
        return {"graph_replays": self.replays, "graph_eager_steps": self.eager_steps,
                "graph_recorded": self.recorded, "graph_growths": self.growths}

    def release(self) -> None:
        """Drop the recorded graph and everything it holds on the device, the
        probes' rings included."""
        self._drop_graph()
        self._warm_key = None
        self._cache = None
        for probe in self.probes:
            probe.drop_ring()

    def _drop_graph(self) -> None:
        # The probes' rings stay: the next recording reuses them.
        self._graph = self._output = None
        self._inputs, self._masks = {}, {}
        self._key = None
        if self._segments is not None:
            self._segments.drop()

    def __call__(self, **kwargs):
        output = self._step({k: v for k, v in kwargs.items() if k not in self.UNUSED})
        self._position += 1
        if self._cache is not None and self._room() <= self._position < self._limit:
            self._grow()
        return output

    def _room(self) -> int:
        return min((layer.max_cache_len for layer in self._cache.layers
                    if type(layer) is StaticLayer), default=self._limit)

    def _grow(self) -> None:
        """Double the answer room between two steps. `generate` sizes the next
        step's mask from the cache, so the next step sees the new length; the
        recording, which reads the old buffers, is dropped."""
        room = min(self._limit, self._prompt + 2 * (self._room() - self._prompt))
        for layer in self._cache.layers:
            if type(layer) is not StaticLayer:
                continue
            if layer.is_initialized:
                for name in ("keys", "values"):
                    old = getattr(layer, name)
                    new = old.new_zeros(*old.shape[:2], room, old.shape[-1])
                    new[:, :, :old.shape[2]].copy_(old)
                    setattr(layer, name, new)
            layer.max_cache_len = room
        self._drop_graph()
        self.growths += 1

    def _step(self, kwargs: dict):
        if self._ring_rows >= self.RING_ROWS:
            for probe in self.probes:
                probe.close_ring()
            return self._eager(kwargs)
        if not self.enabled:
            return self._eager(kwargs)
        tensors = {k: v for k, v in kwargs.items() if torch.is_tensor(v)}
        mask = kwargs.get("attention_mask")
        masks = ({k: v for k, v in mask.items() if torch.is_tensor(v)}
                 if isinstance(mask, dict) else {})
        key = (id(kwargs.get("past_key_values")), self._room(),
               tuple((k, v.shape, v.dtype) for k, v in tensors.items()),
               tuple((k, v.shape, v.dtype) for k, v in masks.items()),
               tuple((k, v) for k, v in kwargs.items()
                     if not torch.is_tensor(v) and k not in ("past_key_values",
                                                             "attention_mask")))
        if key != self._key:
            # A new cache, a grown one, or new shapes: the old recording reads
            # memory that no longer means anything.
            self._drop_graph()
            if key != self._warm_key:
                self._warm_key = key
                return self._eager(kwargs, warm=True)
            if self._segments is not None:
                return self._segment_step(kwargs, key, record=True)
            try:
                self._record(kwargs, tensors, masks, mask)
            except Exception as exc:  # noqa: BLE001 -- any failure means: stay eager
                logger.warning("could not record the decode step into a CUDA "
                               "graph; decoding eagerly from now on: %s: %s",
                               type(exc).__name__, str(exc).split("\n")[0])
                self.enabled = False
                self._drop_graph()
                return self._eager(kwargs)
            self._key = key
        if self._segments is not None:
            return self._segment_step(kwargs, key, record=False)
        for name, value in tensors.items():
            self._inputs[name].copy_(value)
        for name, value in masks.items():
            self._masks[name].copy_(value)
        self._mark_rings()
        self._graph.replay()
        self.replays += 1
        self._ring_rows += 1
        return self._output

    def _eager(self, kwargs: dict, warm: bool = False):
        self.eager_steps += 1
        if self._marked and self._ring_rows < self.RING_ROWS:
            self._ring_rows += 1      # the probes write this step to their rings
        if warm and self._segments is not None and not self._segments.validated:
            output = self._segments.run(self.model, kwargs, "warm")
            problem = self._segments.warm_problem()
            if problem is not None:
                logger.warning("decode graphs: cannot record the decoder layers per "
                               "GPU (%s); decoding eagerly from now on", problem)
                self.enabled = False
            return output
        return self.model(**kwargs)

    def _mark_rings(self) -> None:
        """From the first replayed step on, the probes write to their rings."""
        if not self._marked:
            for probe in self.probes:
                probe.mark_ring()
            self._marked = True

    def _segment_step(self, kwargs: dict, key, record: bool):
        """One decode step through the per-GPU layer recordings (recording
        them first if ``record``). A segment that fails runs eagerly, with the
        rest of the step, and every later step is eager."""
        if record:
            for probe in self.probes:
                probe.use_ring(self.RING_ROWS)
        self._mark_rings()
        output = self._segments.run(self.model, kwargs, "record" if record else "replay")
        self._ring_rows += 1
        failure = self._segments.failure
        if failure is not None:
            logger.warning("could not %s the decode step's per-GPU graphs; decoding "
                           "eagerly from now on: %s: %s",
                           "record" if record else "replay", type(failure).__name__,
                           str(failure).split("\n")[0], exc_info=failure)
            self.enabled = False
            self._drop_graph()
            self.eager_steps += 1
            return output
        self._key = key
        self.recorded |= record
        self.replays += 1
        return output

    def _record(self, kwargs: dict, tensors: dict, masks: dict, mask) -> None:
        for probe in self.probes:
            probe.use_ring(self.RING_ROWS)
        self._inputs = {k: v.clone() for k, v in tensors.items()}
        self._masks = {k: v.clone() for k, v in masks.items()}
        static = dict(kwargs, **self._inputs)
        if isinstance(mask, dict):
            static["attention_mask"] = dict(mask, **self._masks)
        graph = torch.cuda.CUDAGraph()
        # Thread-local: only this thread's own calls can spoil the recording.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            output = self.model(**static)
        self._graph, self._output = graph, output
        self.recorded = True
