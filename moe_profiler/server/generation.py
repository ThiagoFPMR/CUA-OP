"""Turn one OpenAI chat request into one profiled generation.

`openai_to_prompt` converts the request; `profile_generation` runs it through the
hooked model and returns the answer plus its packed routing stats.
"""
from __future__ import annotations

import contextlib
import gc
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from ..encoding import encode_messages
from ..probe import pack_probes

logger = logging.getLogger("moe_profiler.server")


def _split_messages(messages: list[dict]) -> tuple[list[dict], list]:
    """
    Separate an OpenAI chat into (chat with image placeholders, ordered images).

    Each ``{"type": "image_url", "image_url": {"url": ...}}`` part is removed
    from the content and its URL appended to the image list, in the order the
    images appear across the whole conversation — which is the order the
    processor pairs them with the placeholders left behind. String content and
    text parts pass through untouched.
    """
    chat: list[dict] = []
    images: list = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, list):
            # A plain string turn (or empty) carries no images.
            chat.append({"role": role, "content": content})
            continue
        parts: list[dict] = []
        for part in content:
            kind = part.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            elif kind == "image_url":
                images.append(part["image_url"]["url"])
                parts.append({"type": "image"})
            elif kind == "image":
                # Already an engine-style image part: keep the placeholder, and
                # carry its payload (path / bytes / PIL / data URL) positionally.
                images.append(part.get("image"))
                parts.append({"type": "image"})
            else:
                parts.append(part)
        chat.append({"role": role, "content": parts})
    return chat, images


def openai_to_prompt(
    messages: list[dict],
    max_new_tokens: int | None = None,
    chat_template_kwargs: dict | None = None,
) -> dict:
    """
    Build the engine's prompt record from an OpenAI chat-completions request.

    Args:
        messages: The request's ``messages`` — content is a string or a list of
            text / ``image_url`` parts.
        max_new_tokens: The request's ``max_tokens``, carried through as the
            answer's token budget (the engine still clips it to the context
            window). None lets the model run to EOS or the context limit.
        chat_template_kwargs: Options for the chat template, e.g.
            ``{"enable_thinking": True}`` — an agent sends these via the OpenAI
            client's ``extra_body``, which the server sees at the top level.

    Returns:
        A ``{"messages", "images", "max_new_tokens", "chat_template_kwargs"}``
        record, ready for `profile_generation`.
    """
    chat, images = _split_messages(messages)
    return {
        "messages": chat,
        "images": images,
        "max_new_tokens": max_new_tokens,
        "chat_template_kwargs": chat_template_kwargs,
    }


@dataclass
class GenerationResult:
    """One profiled generation: the answer, and the stats that describe it."""

    generated_text: str
    """The model's decoded answer, special tokens stripped."""
    generated_token_ids: list[int]
    """The answer's token ids, in order."""
    prompt_text: str
    """The chat-templated prompt string the model was given."""
    prompt_token_ids: list[int]
    """The prompt's input ids — used to share the long prefix successive steps repeat."""
    arrays: dict[str, np.ndarray]
    """The packed per-token routing stats, ready for `store.write_generation`."""
    n_prompt_input_ids: int
    """Prompt length in input ids."""
    n_prefill_tokens: int
    """Rows the prefill produced; rows before this index are prompt, the rest generated."""
    n_activation_rows: int
    """Total per-token rows: prompt tokens plus the generated tokens fed back in."""
    layout: dict
    """The block layout the stats were written under (expert count, top-k, etc.),
    for the run's metadata. Constant across a model's generations."""
    low_memory: bool = False
    """Whether this generation was served by the low-memory retry after an OOM."""
    timings: dict = field(default_factory=dict)
    """Wall time per phase, in seconds (``*_s``), plus the decode step count.
    Filled here for the model's phases; the server adds its own around them."""


def register_probes(profiler) -> list:
    """
    Attach the profiler's probe hooks and return their handles.

    A long-lived server registers these once and generates every request under
    them; remove the returned handles to detach.
    """
    return [handle for probe in profiler.probes for handle in probe.attach()]


class GenerationAbandoned(RuntimeError):
    """The caller gave up on a generation, which was stopped before it finished."""


class PromptTooLarge(RuntimeError):
    """A prompt the GPU cannot, or is configured not to, generate from.

    Raised both before the model runs (the prompt is over ``max_prompt_tokens``)
    and after a CUDA out-of-memory error, once the memory it held is released.
    Either way the same prompt would fail again: generations never overlap, so
    the memory it needs does not depend on what else is running.
    """


def release_cuda_memory() -> None:
    """Return the memory a failed forward left behind to the device.

    An exception's traceback keeps every frame it unwound through alive — and
    with them the activations those frames held — often inside reference cycles
    only the collector breaks. So collect first, then hand the freed cached
    blocks back, or the next request finds the device still full.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _peak_gib() -> float:
    """The largest peak allocation on any visible GPU since the last reset, GiB."""
    if not torch.cuda.is_available():
        return 0.0
    return max(torch.cuda.max_memory_allocated(d)
               for d in range(torch.cuda.device_count())) / 2**30


def _reset_peaks() -> None:
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)


def _vision_tower(model):
    """The model's vision encoder, if it takes Qwen-VL-style packed patches.

    Only an encoder called as ``visual(pixel_values, grid_thw=...)`` is
    returned: its images arrive as one flat run of patches, image after image,
    that `grid_thw` delimits, so they can be encoded one at a time. Anything
    else is left alone.
    """
    for owner in (model, getattr(model, "model", None)):
        tower = getattr(owner, "visual", None) if owner is not None else None
        if tower is not None and "grid_thw" in inspect.signature(
                tower.forward).parameters:
            return tower
    return None


def _cat_outputs(outputs: list):
    """Concatenate per-image vision outputs back into the batched one."""
    first = outputs[0]
    for key in list(first.keys()):
        values = [o[key] for o in outputs]
        if isinstance(values[0], torch.Tensor):
            first[key] = torch.cat(values)
        elif isinstance(values[0], (list, tuple)) and values[0] \
                and isinstance(values[0][0], torch.Tensor):
            # Per-layer features (e.g. deepstack), each a run of patches.
            first[key] = type(values[0])(torch.cat(parts) for parts in zip(*values))
    return first


@contextlib.contextmanager
def _per_image_vision(model):
    """Encode a prompt's images one at a time instead of as one batch.

    The vision encoder's attention is already per image, so the result is the
    same; what changes is that only one image's activations are alive at once,
    rather than every screenshot in the history. The forward is patched on the
    instance (wrapping whatever accelerate installed there) and restored after.
    """
    tower = _vision_tower(model)
    if tower is None:
        yield
        return
    had_own = "forward" in tower.__dict__
    inner = tower.forward

    def forward(pixel_values, grid_thw=None, **kwargs):
        # Precomputed per-batch tensors (cu_seqlens, ...) would not match a
        # single image's slice; with any of them, run the batch as given.
        if grid_thw is None or len(grid_thw) < 2 or any(
                isinstance(v, torch.Tensor) for v in kwargs.values()):
            return inner(pixel_values, grid_thw=grid_thw, **kwargs)
        sizes = grid_thw.prod(-1).tolist()
        outputs = [inner(pixels, grid_thw=grid, **kwargs)
                   for pixels, grid in zip(torch.split(pixel_values, sizes),
                                           grid_thw.split(1))]
        return _cat_outputs(outputs)

    tower.forward = forward
    try:
        yield
    finally:
        if had_own:
            tower.forward = inner
        else:
            del tower.forward


@contextlib.contextmanager
def _low_memory_mode(model, gen_kwargs: dict, enabled: bool):
    """The retry after an OOM: the same forward, computed in smaller pieces.

    Images go through the vision encoder one at a time, and the attention cache
    is kept on the CPU, each layer's copied to the GPU only while that layer
    runs. Both change where and in what order the work happens, not what is
    computed, so the routing statistics describe the same model on the same
    prompt. Both cost speed, which is why this is a fallback and not the default.
    """
    if not enabled:
        yield
        return
    gen_kwargs["cache_implementation"] = "offloaded"
    try:
        with _per_image_vision(model):
            yield
    finally:
        gen_kwargs.pop("cache_implementation", None)


class _StopWhen(StoppingCriteria):
    """Stop decoding as soon as ``should_stop()`` turns true."""

    def __init__(self, should_stop: Callable[[], bool]):
        self.should_stop = should_stop

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.should_stop(),
                          dtype=torch.bool, device=input_ids.device)


class _TokenClock(StoppingCriteria):
    """Never stops decoding; notes when each token lands.

    Called once per generated token, so its first call marks the end of the
    prefill (the prompt's forward plus the first token) and its last the end of
    decoding. `generate` waits on the device every step to learn whether the
    sequence is finished, so these host timestamps track the GPU to the token.

    The probes' share is the host time spent in their hooks. They never wait on
    the device, so that is the time it takes to launch their work, not to run
    it: their GPU time overlaps the model's and is only in the step totals.
    """

    def __init__(self, probes: list):
        self._probes = probes
        self._start = time.perf_counter()
        self._probe_start = self._probe()
        self._first = self._last = None
        self._first_probe = 0.0
        self._n = 0

    def _probe(self) -> float:
        return sum(p.host_seconds for p in self._probes)

    def __call__(self, input_ids, scores, **kwargs):
        now = time.perf_counter()
        if self._first is None:
            self._first, self._first_probe = now, self._probe()
        self._last = now
        self._n += 1
        return torch.zeros(input_ids.shape[0], dtype=torch.bool,
                           device=input_ids.device)

    def timings(self) -> dict:
        """Prefill and decode wall time, and the probe hooks' host time in each.
        Read right after `generate` returns, before anything else fires a hook."""
        if self._first is None:
            return {}
        return {
            "prefill_s": self._first - self._start,
            "prefill_probe_host_s": self._first_probe - self._probe_start,
            "decode_s": self._last - self._first,
            "decode_probe_host_s": self._probe() - self._first_probe,
            "decode_steps": self._n - 1,
        }


@torch.inference_mode()
def profile_generation(profiler, prompt: dict,
                       should_stop: Callable[[], bool] | None = None,
                       max_prompt_tokens: int | None = None,
                       timings: dict | None = None,
                       decode_graph=None,
                       ) -> GenerationResult:
    """
    Run one prompt through the hooked model and collect its routing statistics.

    The model answers on its own under `generate` (greedy unless the profiler was
    built to sample); the hooks fire on the prefill and every decode step, so the
    packed arrays span the prompt followed by the tokens the model fed back to
    itself. The probe hooks must already be registered — see `register_probes`.

    Args:
        profiler: A `MoEProfiler` whose probes are built and hooked.
        prompt: A ``{"messages", "images", "max_new_tokens",
            "chat_template_kwargs"}`` record, e.g. from `openai_to_prompt`. An
            optional ``min_new_tokens`` keeps the model from stopping before
            that many tokens, which pins the decode length for benchmarking,
            and an optional ``logits_processor`` (a `LogitsProcessorList`)
            steers the answer, e.g. to replay given tokens.
        should_stop: Polled after every decoded token; once it returns True the
            generation is cut short and discarded — its stats would describe a
            truncated answer nobody received.
        max_prompt_tokens: Refuse, before the model runs, a prompt longer than
            this many input ids (image tokens included). None means no cap.
        timings: A dict to fill with each phase's wall time as it completes,
            so a caller holding it sees how far a generation got even when it
            raises. Becomes the result's ``timings``.
        decode_graph: A `decode_graph.DecodeGraph` on this model, to replay the
            decode steps from a CUDA graph. The generation then runs on the
            graph's static cache; the low-memory retry never does. None decodes
            eagerly on the default cache.

    Returns:
        The `GenerationResult` for this request.

    Raises:
        GenerationAbandoned: ``should_stop`` fired before the answer finished.
        PromptTooLarge: The prompt is over ``max_prompt_tokens``, or the GPU ran
            out of memory on it both as-is and in the low-memory retry (the
            memory is released before this is raised).
        RuntimeError: if any hooked MoE block never fired during the forward —
            the routing statistics would be incomplete.
    """
    timings = {} if timings is None else timings
    started = time.perf_counter()
    enc, prompt_text = encode_messages(
        profiler.processor, prompt["messages"], prompt["images"],
        profiler.device, prompt.get("chat_template_kwargs"))
    timings["encode_s"] = time.perf_counter() - started
    n_tok = enc["input_ids"].shape[1]
    n_images = len(prompt["images"])
    if max_prompt_tokens and n_tok > max_prompt_tokens:
        raise PromptTooLarge(
            f"prompt is {n_tok} tokens ({n_images} images), over "
            f"max_prompt_tokens={max_prompt_tokens}")

    probes = profiler.probes
    for probe in probes:
        probe.reset()

    gen_kwargs = {"do_sample": profiler.do_sample, "num_beams": 1, "use_cache": True}
    budgets = [b for b in (
        max(profiler.context_window - n_tok, 1) if profiler.context_window else None,
        prompt.get("max_new_tokens"),
    ) if b]
    if budgets:
        gen_kwargs["max_new_tokens"] = min(budgets)
    if prompt.get("min_new_tokens"):
        gen_kwargs["min_new_tokens"] = prompt["min_new_tokens"]
    if prompt.get("logits_processor") is not None:
        gen_kwargs["logits_processor"] = prompt["logits_processor"]

    stops = [_StopWhen(should_stop)] if should_stop is not None else []

    # Tried as-is first. Only a prompt that runs the GPU out of memory is retried
    # in the low-memory mode, which computes the same forward in smaller pieces
    # at some cost in speed.
    sequences, low_memory = None, False
    generating, generating_cpu = time.perf_counter(), time.thread_time()
    for low_memory in (False, True):
        _reset_peaks()
        # A fresh clock per attempt: after an OOM retry, prefill and decode are
        # the retry's; the failed attempt shows only in the total.
        clock = _TokenClock(probes)
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([*stops, clock])
        graphed = (decode_graph is not None and not low_memory
                   and "max_new_tokens" in gen_kwargs)
        if graphed:
            # The graph's own static cache is what routes `generate`'s decode
            # steps through it (see decode_graph).
            gen_kwargs["past_key_values"] = decode_graph.start(
                n_tok, gen_kwargs["max_new_tokens"])
        oom = None
        try:
            with _low_memory_mode(profiler.model, gen_kwargs, enabled=low_memory):
                sequences = profiler.model.generate(**enc, **gen_kwargs)
        except torch.OutOfMemoryError as exc:
            # Keep only the message: holding the exception past this block
            # would keep its traceback, and every activation it pins, alive.
            oom = str(exc).split(". ")[0]
        finally:
            if graphed:
                gen_kwargs.pop("past_key_values", None)
                timings.update(decode_graph.end())
        if oom is None:
            break
        for probe in probes:
            probe.reset()
        release_cuda_memory()
        logger.error("out of memory on a %d-token prompt (%d images)%s: %s",
                     n_tok, n_images, " in low-memory mode" if low_memory else
                     "; retrying in low-memory mode", oom)
        if should_stop is not None and should_stop():
            raise GenerationAbandoned("the caller gave up mid-generation")
    if sequences is None:
        enc = None
        raise PromptTooLarge(
            f"out of GPU memory on a {n_tok}-token prompt ({n_images} images), "
            f"even in low-memory mode; send fewer or smaller images")
    timings.update(clock.timings(),
                   generate_s=time.perf_counter() - generating,
                   generate_cpu_s=time.thread_time() - generating_cpu)
    logger.info("generated %d tokens from a %d-token prompt (%d images)%s; "
                "peak GPU memory %.1f GiB",
                sequences.shape[1] - n_tok, n_tok, n_images,
                " in low-memory mode" if low_memory else "", _peak_gib())
    if should_stop is not None and should_stop():
        raise GenerationAbandoned("the caller gave up mid-generation")
    packing = time.perf_counter()
    for probe in probes:
        probe.finalize()
    missing = [i for i, p in enumerate(probes) if p.top_idx is None]
    if missing:
        raise RuntimeError(
            f"{len(missing)} MoE blocks never fired during the forward "
            f"(first: block {missing[0]}). The forward may route around them; "
            f"check the profiler's MoE scope.")

    gen_ids = sequences[0, n_tok:].tolist()
    gen_text = profiler.tokenizer.decode(gen_ids, skip_special_tokens=True)

    arrays = pack_probes(probes)
    prompt_ids = enc["input_ids"][0].tolist()
    timings["pack_s"] = time.perf_counter() - packing

    return GenerationResult(
        generated_text=gen_text,
        generated_token_ids=gen_ids,
        prompt_text=prompt_text,
        prompt_token_ids=prompt_ids,
        arrays=arrays,
        n_prompt_input_ids=n_tok,
        n_prefill_tokens=probes[0].chunk_sizes[0],
        n_activation_rows=probes[0].sim.shape[0],
        layout={
            "n_experts": probes[0].n_experts,
            "top_k": probes[0].n_selected,
            "expert_layout": probes[0].layout,
            "scores_scattered": probes[0].scores_scattered,
            "stats_top_m": min(profiler.top_m, probes[0].n_experts),
        },
        low_memory=low_memory,
        timings=timings,
    )
