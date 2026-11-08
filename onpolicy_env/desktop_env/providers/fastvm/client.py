"""`fastvm.FastvmClient`, with a clear error when the `fastvm` SDK is not installed."""
from __future__ import annotations

from functools import lru_cache

try:
    from fastvm import FastvmClient
except ImportError as e:  # pragma: no cover — install-time hint
    raise ImportError(
        "The `fastvm` Python SDK is required for the FastVM OSWorld provider.\n"
        "Install it with: pip install fastvm\n"
        "Or, for local development: pip install -e /path/to/fastvm-sdk-python"
    ) from e


@lru_cache(maxsize=1)
def get_client() -> FastvmClient:
    """Return a process-wide FastvmClient.

    `FastvmClient` reads `FASTVM_API_KEY` and `FASTVM_BASE_URL` from the
    environment on construction. One client per process is plenty — it
    manages its own HTTP/2 connection pool.
    """
    return FastvmClient()
