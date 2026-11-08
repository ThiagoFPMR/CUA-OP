"""ProfilingClient: start, step and end episodes on the profiling server.

The agent sends its chat requests itself, to `episode_base_url(episode_id)`.
"""
from __future__ import annotations

import requests


class ProfilingClient:
    """Open, advance and close episodes on a running profiling server."""

    def __init__(self, server_url: str, model_path: str, runs_root: str,
                 timeout: float = 30.0, start_timeout: float = 1800.0):
        """
        Args:
            server_url: The server's root, e.g. ``http://gpu-box:8000``.
            model_path: The checkpoint to profile, as a path on the server.
            runs_root: The directory on the server the episodes' runs go under.
            timeout: Seconds to wait on a step or an end.
            start_timeout: Seconds to wait on a start, which may make the
                server load the checkpoint first.
        """
        self.base = server_url.rstrip("/")
        self.model_path = model_path
        self.runs_root = runs_root
        self.timeout = timeout
        self.start_timeout = start_timeout

    def episode_base_url(self, episode_id: str) -> str:
        """The OpenAI base URL an agent uses to reach this episode."""
        return f"{self.base}/ep/{episode_id}/v1"

    def start(self, episode_id: str, task_id: str | None, domain: str | None) -> None:
        self._post("/episode/start", {"episode_id": episode_id,
                                      "model_path": self.model_path,
                                      "runs_root": self.runs_root,
                                      "task_id": task_id, "domain": domain},
                   timeout=self.start_timeout)

    def step(self, episode_id: str, step: int, stage: str | None = None) -> None:
        self._post(f"/ep/{episode_id}/step", {"step": step, "stage": stage})

    def end(self, episode_id: str, stop_reason: str, score) -> None:
        self._post(f"/ep/{episode_id}/end",
                   {"stop_reason": stop_reason, "score": score})

    def _post(self, path: str, body: dict, timeout: float | None = None) -> dict:
        resp = requests.post(self.base + path, json=body,
                             timeout=timeout or self.timeout)
        resp.raise_for_status()
        return resp.json()
