"""Load a MoE VLM checkpoint plus its input processor."""
import torch

from transformers import AutoModelForImageTextToText, AutoProcessor


def load_model_and_processor(model_path: str, attn: str = "eager"):
    """Load a MoE VLM checkpoint plus its input processor."""
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation=attn)
    model.eval()
    print(f"[load] model_type={getattr(model.config, 'model_type', '?')}", flush=True)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"[load] processor={type(processor).__name__}", flush=True)
    return model, processor


def text_config(model):
    """The sub-config carrying the MoE hyperparameters (VLMs nest it)."""
    cfg = model.config
    for attr in ("text_config", "llm_config", "language_config"):
        sub = getattr(cfg, attr, None)
        if sub is not None and getattr(sub, "num_experts_per_tok", None) is not None:
            return sub
    return getattr(cfg, "text_config", cfg)
