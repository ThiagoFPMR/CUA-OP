"""Encode one generation's prompt: chat-templated messages plus their images."""
import base64
import io

import torch
from PIL import Image
from transformers import AutoProcessor


def _load_image(src) -> Image.Image:
    """
    Load one image, whatever form it arrives in, as an RGB `PIL.Image`.

    A prompt may name its images as filesystem paths (a set read from disk), as
    already-decoded `PIL.Image` objects, as raw encoded bytes, or as a ``data:``
    URL carrying base64-encoded bytes (how a screenshot travels inside an
    OpenAI-vision chat request). All four resolve to the same pixels here.
    """
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(src))).convert("RGB")
    if isinstance(src, str) and src.startswith("data:"):
        # data:[<mediatype>][;base64],<payload>
        payload = src.partition(",")[2]
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    return Image.open(src).convert("RGB")


def encode_messages(
    processor: AutoProcessor,
    messages: list[dict],
    images: list,
    device: torch.device,
    chat_template_kwargs: dict | None = None,
) -> tuple[dict, str]:
    """
    Encode one generation's prompt -- chat-templated messages plus its images
    -- ready for generation.

    Args:
        processor: The AutoProcessor to use for encoding.
        messages: The chat to answer; the template adds the generation prompt.
        images: The images those messages reference, in order, each a
            filesystem path, a `PIL.Image`, raw encoded bytes, or a ``data:``
            URL. See `_load_image`.
        device: The device to move the encoded tensors to.
        chat_template_kwargs: Options for the chat template, e.g.
            ``{"enable_thinking": True}``. They are forwarded into the template's
            own environment, so a template that does not declare one ignores it.

    Returns:
        (enc, prompt_text): the encoded tensors on `device`, and the
        chat-templated prompt string.
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        **(chat_template_kwargs or {}))
    kwargs = {"text": [text], "return_tensors": "pt"}
    if images:
        kwargs["images"] = [_load_image(s) for s in images]
    enc = processor(**kwargs)
    enc = {k: v.to(device) if hasattr(v, "to") else v for k, v in enc.items()}
    return enc, text
