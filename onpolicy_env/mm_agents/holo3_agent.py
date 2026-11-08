"""Holo3Agent: an OSWorld agent for a Holo3 model behind an OpenAI-compatible API.

Each step is one model call answering JSON {note, thought, tool_call}. The server
URL is reread from OPENAI_BASE_URL on every call; the message format is in
`holo3_format`.
"""
import base64
import logging
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import openai
from PIL import Image

from onpolicy_env.mm_agents.holo3_format import (
    HISTORY_LENGTH,
    IMAGE_BUDGET,
    LOOP_MAX_TOKENS,
    build_agent_loop_conversation,
    parse_json,
    replayed_answer,
)

logger = logging.getLogger("desktopenv.holo3_agent")

MAX_RETRY_TIMES = 3


def json_schema_response_format(model_cls, name: str) -> dict:
    """OpenAI/SGLang guided-decoding format that forces the model to emit a schema.

    NOT used by the agent loop -- CUAPruning profiled the model with unconstrained
    decoding, so the served agent must too, or it benchmarks a distribution the
    model was never pruned under. Unused here; kept for a standalone grounding
    check, which may want to force a clean Click(x, y) out of a bare endpoint.
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": model_cls.model_json_schema(), "strict": True},
    }


# Holo click_type -> pyautogui pointer function.
_CLICK_FN = {
    "left": "click", "double": "doubleClick", "right": "rightClick",
    "middle": "middleClick", "triple": "tripleClick", "move": "moveTo",
}


class Holo3Agent:
    def __init__(
        self,
        platform: str = "ubuntu",
        model: str = "holo3-gimp-pruned",
        max_tokens: int = LOOP_MAX_TOKENS,
        top_p: float = 0.95,
        temperature: float = 0.0,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot",
        coord_divisor: Optional[float] = None,
        scroll_clicks: int = 3,
        invert_scroll: bool = False,
        history_length: int = HISTORY_LENGTH,
        image_budget: int = IMAGE_BUDGET,
        enable_thinking: bool = True,
        max_parse_retries: int = 3,
        retry_temperature: float = 0.0,
    ):
        self.platform = platform
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        # Model coordinate space. H's hub documents [0, 1000]; env overrides so a
        # grounding smoke test can retune without a rebuild.
        self.coord_divisor = float(
            coord_divisor if coord_divisor is not None
            else os.environ.get("HOLO_COORD_DIVISOR", "1000"))
        self.scroll_clicks = scroll_clicks
        # pyautogui.scroll is "positive = up"; flip if a trial scrolls the wrong way.
        self.invert_scroll = invert_scroll
        self.history_length = history_length
        # How many screenshots survive as images; older <observation>s keep their
        # wrapper but their pixels become a text placeholder. Matches CUAPruning.
        self.image_budget = image_budget
        # The agent loop plans before acting; the <think> trace is stripped before
        # parsing. Enabled on the server via chat_template_kwargs.
        self.enable_thinking = enable_thinking
        self.max_parse_retries = max(1, max_parse_retries)
        self.retry_temperature = retry_temperature

        assert action_space == "pyautogui", "Holo3Agent only supports pyautogui"
        assert observation_type == "screenshot", "Holo3Agent only supports screenshot"

        self._reset_memory()

    def _reset_memory(self) -> None:
        # One entry per completed step, oldest first. Each carries the screenshot
        # the model saw (base64 PNG), the assistant JSON it emitted (parsed and
        # re-dumped, thinking stripped), and the tool it called. This is exactly
        # what build_agent_loop_conversation replays into the next prompt.
        self.history: List[dict] = []
        self.responses: List[str] = []         # raw generation per step, for the traj log

    # ------------------------------------------------------------------ predict
    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        png = obs["screenshot"]
        width, height = Image.open(BytesIO(png)).size
        b64 = base64.b64encode(png).decode("utf-8")

        # Rebuild the multi-turn conversation the served agent would be holding:
        # system(task) + replayed past steps + the current <observation>.
        messages = build_agent_loop_conversation(
            task=instruction,
            current_image=b64,
            history=self.history,
            history_length=self.history_length,
            image_budget=self.image_budget,
        )
        messages = self._inline_images(messages)

        raw, step = self._generate_step(messages)
        self.responses.append(raw)

        if step is None or not isinstance(step.get("tool_call"), dict):
            logger.error("Holo3 agent loop produced no usable tool_call.")
            # Nothing to replay for a failed step: leave history untouched so the
            # next prompt does not claim a turn that never resolved.
            return raw, []

        pyautogui_code, log = self._translate(step["tool_call"], width, height)

        # Record the step for replay -- the parsed, re-dumped answer, never the
        # raw generation or its <think> trace.
        answer, tool_name = replayed_answer(raw)
        self.history.append({"image": b64, "answer": answer, "tool_name": tool_name})

        logger.info("Holo3 step: %s -> %s", log, pyautogui_code)
        return raw, pyautogui_code

    def _generate_step(self, messages: List[dict]) -> Tuple[str, Optional[dict]]:
        """Call the loop, parse; re-sample on a parse miss up to max_parse_retries."""
        raw = ""
        for attempt in range(self.max_parse_retries):
            temperature = self.temperature if attempt == 0 else self.retry_temperature
            # A zero retry temperature would resample identically, so stop early.
            if attempt > 0 and self.retry_temperature <= 0.0:
                break
            raw = self.call_llm(messages, self.max_tokens, temperature=temperature)
            step = parse_json(raw)
            if isinstance(step, dict) and isinstance(step.get("tool_call"), dict):
                return raw, step
            logger.warning("Holo3 parse miss (attempt %d/%d).", attempt + 1,
                           self.max_parse_retries)
        return raw, parse_json(raw)

    # ---------------------------------------------------------------- translate
    def _translate(self, call: dict, width: int, height: int) -> Tuple[List[str], str]:
        """Holo tool_call dict -> (pyautogui code list, log string).

        Coordinates arrive inline in Holo's model space.
        """
        name = call.get("tool_name")

        def to_px(xm, ym) -> Tuple[int, int]:
            x = round(float(xm) / self.coord_divisor * width)
            y = round(float(ym) / self.coord_divisor * height)
            return max(0, min(width - 1, x)), max(0, min(height - 1, y))

        if name == "click":
            if "x" not in call or "y" not in call:
                return [], "click (no coordinates)"
            xy = to_px(call["x"], call["y"])
            fn = _CLICK_FN.get(call.get("click_type", "left"), "click")
            return [f"pyautogui.{fn}({xy[0]}, {xy[1]})"], f"{fn} @ {xy}"

        if name == "drag":
            if "x" not in call or "y" not in call:
                return [], "drag (no coordinates)"
            xy = to_px(call["x"], call["y"])
            button = call.get("button", "left")
            return [f"pyautogui.dragTo({xy[0]}, {xy[1]}, button={button!r}, duration=0.5)"], \
                f"drag @ {xy}"

        if name == "write":
            content = call.get("content", "")
            code = [f"pyautogui.write({content!r}, interval=0.02)"]
            if call.get("press_enter"):
                code.append("pyautogui.press('enter')")
            return code, f"write {content!r}" + (" +enter" if call.get("press_enter") else "")

        if name == "press_keys":
            keys = call.get("keys", [])
            keys = [keys] if isinstance(keys, str) else list(keys)
            if call.get("hold"):
                inner = ", ".join(repr(k) for k in keys)
                return [f"pyautogui.hotkey({inner})"], f"hotkey {keys}"
            if len(keys) == 1:
                presses = int(call.get("presses", 1) or 1)
                code = f"pyautogui.press({keys[0]!r}" + (f", presses={presses})" if presses > 1 else ")")
                return [code], f"press {keys[0]} x{presses}"
            return [f"pyautogui.press({keys!r})"], f"press {keys}"

        if name == "scroll":
            direction = call.get("direction", "down")
            mag = self.scroll_clicks
            if direction in ("up", "down"):
                amt = mag if direction == "up" else -mag
                if self.invert_scroll:
                    amt = -amt
                return [f"pyautogui.scroll({amt})"], f"scroll {direction}"
            amt = mag if direction == "right" else -mag
            return [f"pyautogui.hscroll({amt})"], f"hscroll {direction}"

        if name == "wait":
            return ["WAIT"], "wait"

        if name == "answer":
            status = call.get("status", "success")
            return (["DONE"] if status == "success" else ["FAIL"]), f"answer {status}"

        logger.error("Unknown Holo3 tool_call: %r", call)
        return [], f"unknown {name}"

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _inline_images(messages: List[dict]) -> List[dict]:
        """Swap holo3_format's carrier image items for OpenAI data-URL parts.

        Each image part carries its own base64 PNG (set by _image_item), so a
        multi-image history inlines the right screenshot per <observation> --
        evicted ones were already turned into text by the trim step.
        """
        out = []
        for m in messages:
            content = m["content"]
            if isinstance(content, list):
                new = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        b64 = item.get("image", "")
                        data_url = f"data:image/png;base64,{b64}"
                        new.append({"type": "image_url", "image_url": {"url": data_url}})
                    else:
                        new.append(item)
                out.append({"role": m["role"], "content": new})
            else:
                out.append(m)
        return out

    # Kept for the grounding smoke test, which builds a one-off localization call.
    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        return parse_json(text)

    def call_llm(self, messages: List[dict], max_tokens: int,
                 temperature: Optional[float] = None,
                 response_format: Optional[dict] = None) -> str:
        base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        # Retry only in the loop below; the SDK's own retries would re-send
        # each attempt unseen. See mm_agents/qwen/client.py for the timeout.
        client = openai.OpenAI(
            base_url=base_url, api_key=api_key, max_retries=0,
            timeout=float(os.environ.get("OSWORLD_OPENAI_TIMEOUT", "1800")))
        kwargs: Dict[str, Any] = dict(
            model=self.model, messages=messages, max_tokens=max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p)
        # Thinking is a chat-template option, forwarded to SGLang/vLLM via
        # extra_body -- the same knob CUAPruning set as chat_template_kwargs.
        if self.enable_thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        if response_format is not None:
            kwargs["response_format"] = response_format
        for attempt in range(1, MAX_RETRY_TIMES + 1):
            try:
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                logger.error("[Holo3] call failed (attempt %d/%d): %s", attempt, MAX_RETRY_TIMES, e)
                if attempt < MAX_RETRY_TIMES:
                    time.sleep(5)
        return ""

    # ------------------------------------------------------------------- reset
    def reset(self, _logger=None, vm_ip=None, **kwargs):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.holo3_agent")
        self.vm_ip = vm_ip
        self._reset_memory()
