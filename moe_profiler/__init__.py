"""Record per-token MoE routing statistics from a Hugging Face model.

`profiler.MoEProfiler` hooks a model's MoE blocks, `store` defines the saved stats,
`run.Run` reads saved runs, and `server` serves the hooked model over HTTP.
"""
from importlib import import_module

# Loaded on first use: the hooks need torch and transformers, but a client of
# the server (`server.client`) or a reader of its runs (`run`, `store`) needs
# neither, and must not have to install them.
_EXPORTS = {
    "MoEProfiler": ".profiler",
    "load_model_and_processor": ".loading",
    "text_config": ".loading",
    "read_generation": ".store",
    "read_outputs": ".store",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
