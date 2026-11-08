"""The messages the Holo3 agent sends: system prompt, answer schema, and history.

Only the last `image_budget` screenshots stay images; older ones become text
placeholders. Changing this format changes what the profiled model reads.
"""
import json
import re
from typing import Any, Literal, Union

from pydantic import BaseModel, Field

# --- The agent loop, desktop-adapted from hub.hcompany.ai/agent-loop -------------
# Verbatim from CUAPruning/calibration/models/holo3.py: the embedded schema and
# every field description below are rendered into the system prompt, so any
# divergence changes the bytes the model reads.

SYSTEM_PROMPT: str = """Imagine you are a robot operating a desktop computer, just like a human. Now you need to complete a task.
In each iteration you receive an observation holding a screenshot of the current screen.
Carefully analyze the visual information to identify what to do, then follow the guidelines to choose the next tool call.
Detail your reasoning in the thought field before acting, and record in the note field anything on the screen that later steps will need.

Guidelines:
- Use the task and the current screenshot to decide the single next tool call.
- The note field is your memory across steps: put paths, names, counts and intermediate answers there. Set it to null when the screen shows nothing new worth recording.
- Name the target in the element field by its visible name, label, icon, or distinguishing features (shape, colour, position), and place it with the x/y fields, as integers in [0, 1000] measured from the left and top edges of the screenshot.
- To type into a field, make sure it is focused first (click it), then use the write tool.
- Consolidate repeated key presses using the presses field rather than emitting many press_keys calls.
- For window controls, identify them correctly: minimize is the '-', maximize the square, close the 'X'.
- Only answer with status 'success' when the screenshot confirms the task is complete; use 'failure' when it cannot be completed.

<task>
{task}
</task>

<output_format>
```json
{output_format}
```
</output_format>
"""


class ClickTool(BaseModel):
    """Click a desktop UI element identified by its description"""

    tool_name: Literal["click"] = "click"
    element: str = Field(description="Detailed description of the target UI element to click on")
    x: int = Field(description="X coordinate as integer in [0, 1000]")
    y: int = Field(description="Y coordinate as integer in [0, 1000]")
    click_type: Literal["left", "double", "right", "middle", "triple", "move"] = Field(
        default="left", description="Which click to perform")


class DragTool(BaseModel):
    """Drag from the current cursor position to a desktop element"""

    tool_name: Literal["drag"] = "drag"
    element: str = Field(description="Detailed description of the drop target")
    x: int = Field(description="X coordinate as integer in [0, 1000]")
    y: int = Field(description="Y coordinate as integer in [0, 1000]")
    button: Literal["left", "right", "middle"] = Field(
        default="left", description="Mouse button to hold during the drag")


class WriteTool(BaseModel):
    """Type text into the currently focused element without clicking first"""

    tool_name: Literal["write"] = "write"
    content: str = Field(description="Content to write")
    press_enter: bool = Field(default=False, description="Whether to press Enter after typing")


class PressKeysTool(BaseModel):
    """Press a key, a chord of keys, or a key repeated several times"""

    tool_name: Literal["press_keys"] = "press_keys"
    keys: list[str] = Field(description="Key names, e.g. ['enter'] or ['ctrl', 'c']")
    hold: bool = Field(default=False, description="True = press all keys together (hotkey chord)")
    presses: int = Field(default=1, description="Number of times to press (for a single key)")


class ScrollTool(BaseModel):
    """Scroll in a direction"""

    tool_name: Literal["scroll"] = "scroll"
    direction: Literal["up", "down", "left", "right"] = Field(
        default="down", description="The direction to scroll in")


class WaitTool(BaseModel):
    """Wait briefly for the UI to update"""

    tool_name: Literal["wait"] = "wait"


class AnswerTool(BaseModel):
    """Provide a final answer: the task is complete or cannot be completed"""

    tool_name: Literal["answer"] = "answer"
    status: Literal["success", "failure"] = Field(
        default="success", description="Whether the task was completed")
    content: str = Field(default="", description="The answer content")


ToolCall = Union[
    ClickTool, DragTool, WriteTool, PressKeysTool, ScrollTool, WaitTool, AnswerTool,
]


class Step(BaseModel):
    """One iteration of the agent loop: the object Holo emits per step."""

    note: str | None = Field(
        default=None,
        description="Task-relevant information from the previous observation. Empty if nothing new.")
    thought: str = Field(description="Reasoning about next steps")
    tool_call: ToolCall = Field(discriminator="tool_name")


# --- Constants, kept identical to CUAPruning ------------------------------------

#: Holo's own coordinate space, integers in [0, 1000]. The serving agent scales
#: by this to reach pixels.
COORD_DIVISOR = 1000.0

#: Per-stage generation caps. The agent loop answers a {note, thought, tool_call}
#: JSON *after* a reasoning pass, so it gets the documented output ceiling of
#: holo3-1-35b-a3b; localization answers a two-number click with thinking off.
#: These are the caps the model was profiled/pruned under -- giving it more here
#: would benchmark a distribution it was never served on.
LOOP_MAX_TOKENS = 4096
LOC_MAX_TOKENS = 256

#: How many screenshots stay in context. H documents keeping the last 3 and
#: replacing older ones with a text placeholder inside their own <observation>
#: wrapper; more images degrade accuracy.
IMAGE_BUDGET = 3
EVICTED_TEXT = "[screenshot evicted]"

#: What a replayed <tool_output> turn carries. The layout is documented, its
#: desktop payload is not; every landed step is followed by the screenshot it
#: produced, so the turn is a bare success acknowledgement and nothing more.
TOOL_OUTPUT = "success"

#: How many past steps the agent loop replays. Matches the serving agent's
#: default. The image budget above is separate and applies on top: earlier
#: observations stay in the conversation as text once their screenshot is evicted.
HISTORY_LENGTH = 6


def _image_item(image: Any) -> dict[str, Any]:
    """A content part carrying an image.

    ``image`` is an opaque reference the serving agent knows how to inline (here,
    the base64 PNG of one screenshot). On the wire H documents
    ``{"type": "image_url", "image_url": {"url": ...}}``; holo3_agent swaps to
    that shape at send time, so only the carrier differs, never the layout.
    """
    return {"type": "image", "image": image}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _observation(image: Any) -> dict:
    """One <observation> user turn: the documented wrapper around a screenshot."""
    return {"role": "user", "content": [
        {"type": "text", "text": "<observation>\n"},
        _image_item(image),
        {"type": "text", "text": "\n</observation>"},
    ]}


def _tool_output(tool_name: str) -> dict:
    """The user turn that carries a tool's result back into the loop.

    A ``user`` message, not a ``tool``-role one: H lists sending it as a ``tool``
    role among the ways a structured-output loop goes wrong.
    """
    return {"role": "user", "content": [{"type": "text", "text":
            f'<tool_output tool="{tool_name}">\n{TOOL_OUTPUT}\n</tool_output>'}]}


def _system(task: str) -> dict:
    system_prompt = SYSTEM_PROMPT.format(
        task=task, output_format=json.dumps(Step.model_json_schema()))
    return {"role": "system", "content": [{"type": "text", "text": system_prompt}]}


def build_agent_loop_messages(task: str, image: Any) -> list[dict]:
    """The first turn of the loop: system prompt + the opening observation.

    Verbatim shape from CUAPruning's ``build_agent_loop_messages``. For step N > 0
    use :func:`build_agent_loop_conversation`, which splices the replayed history
    between the system turn and the current observation.
    """
    return [_system(task), _observation(image)]


def build_agent_loop_conversation(
    task: str,
    current_image: Any,
    history: list[dict],
    history_length: int = HISTORY_LENGTH,
    image_budget: int = IMAGE_BUDGET,
) -> list[dict]:
    """Rebuild the full multi-turn agent-loop prompt the served agent would hold.

    This is the inference-time counterpart of CUAPruning's ``HoloModel._replay``
    + ``_trim_images``, built forward from the agent's own memory instead of
    reconstructed from calibration Turn records -- but it renders the identical
    conversation.

    ``history`` is a list of past-step dicts, oldest first, each::

        {"image": <ref>,        # the screenshot the model saw that step
         "answer": <str>,       # the assistant JSON it emitted (parsed+re-dumped)
         "tool_name": <str>}    # the tool it called, naming the <tool_output> turn

    Only the last ``history_length`` steps are replayed; then the whole message
    list is trimmed to the last ``image_budget`` screenshots.
    """
    messages: list[dict] = [_system(task)]

    past = history[-history_length:] if history_length > 0 else []
    for entry in past:
        messages.append(_observation(entry["image"]))
        # The parsed step, never the raw generation and never the reasoning: H
        # calls out replaying raw output as the cause of a whole run collapsing
        # after one bad step.
        messages.append(_assistant(entry["answer"]))
        tool_name = entry.get("tool_name") or ""
        # ``answer`` ends the episode, so it is the one tool whose result never
        # comes back -- the documented loop returns before appending.
        if tool_name and tool_name != "answer":
            messages.append(_tool_output(tool_name))

    messages.append(_observation(current_image))
    _trim_images(messages, image_budget)
    return messages


def _parts(message: dict) -> list[dict]:
    """A message's content as a list of parts, whatever shape it arrived in."""
    content = message.get("content")
    return list(content) if isinstance(content, list) else []


def _is_image(part: dict) -> bool:
    """Whether a content part carries an image, in either carrier."""
    return isinstance(part, dict) and part.get("type") in ("image", "image_url")


def _trim_images(messages: list[dict], image_budget: int) -> None:
    """Keep only the last ``image_budget`` screenshots, in place.

    A port of the trim helper H documents (CUAPruning's ``_trim_images``): older
    image parts become a short text placeholder while their <observation> wrapper
    stays, so the model still sees that a step happened there.
    """
    if image_budget < 0:
        return
    parts = [part for m in messages if m.get("role") == "user"
             for part in _parts(m) if _is_image(part)]
    if len(parts) <= image_budget:
        return
    keep_from = len(parts) - image_budget
    for part in parts[:keep_from]:
        part.clear()
        part["type"] = "text"
        part["text"] = EVICTED_TEXT


# --- Reading generations back ---------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Drop the reasoning channel, keeping only the action.

    The agent loop runs with thinking on, and H is explicit that the trace is for
    visibility only: it never goes back into the conversation, and the action
    lives in ``content``. Holo's chat template emits the opening ``<think>``
    itself as part of the generation prompt, so a real answer starts *inside* the
    block and closes it. The trace has to come off before the JSON can be read,
    or a brace inside the reasoning captures the lenient extraction below.
    """
    if not text:
        return text
    out = _THINK_RE.sub("", text)
    if "</think>" in out:      # opening tag emitted by the template, not the model
        out = out.rsplit("</think>", 1)[1]
    if "<think>" in out:       # unterminated: the answer never arrived
        out = out.split("<think>", 1)[0]
    return out


def parse_json(text: str) -> dict | None:
    """Lenient JSON extraction: whole string, fenced block, or first {...}.

    Generations are not grammar-constrained -- CUAPruning profiled the model with
    plain ``generate`` and no constrained decoder, so the schema that matters is
    the copy living in the system prompt, and this matches that regime -- so
    being forgiving here reads the model's *decision* rather than its punctuation.
    """
    text = strip_thinking(text)
    if not text or not text.strip():
        return None
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    for candidate in (text, *fenced):
        try:
            return json.loads(candidate.strip())
        except Exception:  # noqa: BLE001
            continue
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return None
    return None


def replayed_answer(text: str) -> tuple[str, str]:
    """One assistant turn, re-dumped from what it was parsed into.

    Returns ``(json_to_replay, tool_name)``. The tool name names the
    <tool_output> turn that follows it. Mirrors CUAPruning's ``_replayed_answer``:
    the raw generation and its <think> trace never re-enter the conversation.
    """
    obj = parse_json(text)
    if obj is None:
        return strip_thinking(text).strip(), ""
    call = obj.get("tool_call")
    tool_name = str(call.get("tool_name") or "") if isinstance(call, dict) else ""
    try:
        return json.dumps(Step.model_validate(obj).model_dump()), tool_name
    except Exception:  # noqa: BLE001
        return json.dumps(obj), tool_name


# --- Localization: verbatim from hub.hcompany.ai/element-localization ------------
# Not part of the agent loop -- Holo grounds inline there, and nothing in this
# repo calls this. Kept as the standalone grounding primitive, for pinning down
# the coordinate convention before a run.

class LocalizerOutput(BaseModel):
    """The grounding answer: a click position in Holo's [0, 1000] space."""

    x: int = Field(ge=0, le=1000, description="X coordinate as integer in [0, 1000]")
    y: int = Field(ge=0, le=1000, description="Y coordinate as integer in [0, 1000]")


LOCALIZATION_TASK_PROMPT = (
    "Localize an element on the GUI image according to the provided target "
    "and output a click position.\n"
    " * You must output a valid JSON following the format: {schema}\n"
    " Your target is:\n{element}"
)


def build_localization_messages(instruction: str, image: Any) -> list[dict]:
    # One user turn, image first: no system turn, and the schema inlined in the
    # prompt itself, exactly as H's grounding example builds it.
    prompt = LOCALIZATION_TASK_PROMPT.format(
        schema=LocalizerOutput.model_json_schema(), element=instruction)
    return [
        {"role": "user", "content": [
            _image_item(image),
            {"type": "text", "text": prompt},
        ]},
    ]
