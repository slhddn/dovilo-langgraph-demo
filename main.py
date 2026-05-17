"""
Dovilo webhooks <-> agentic LangGraph workflow.

Architecture:
  Dovilo Desktop ──HTTP webhook──▶ FastAPI :8000/webhook   (incoming)
         ▲
         └── stdio MCP ── this app spawns dovilo-mcp.exe and exposes its
                          tool schemas to Claude as `tools`. Claude calls
                          them itself via tool_use.

The orchestrator's only job is to receive (verified) webhooks and wake the
agent. The agent owns the task lifecycle — it decides when to add notes,
mark steps, fetch full task body via get_task, and finalize the task with
update_status. The host process does not intervene.

Webhook security follows the Dovilo spec: HMAC-SHA256 over `{t}.{rawBody}`,
5-minute replay window, deduplication via X-Dovilo-Delivery.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, Optional, TypedDict

import uvicorn
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from langgraph.graph import END, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("dovilo-demo")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DOVILO_MCP_CONFIG = Path(os.environ.get("DOVILO_MCP_CONFIG", ".mcp.json"))
DOVILO_MCP_SERVER_NAME = os.environ.get("DOVILO_MCP_SERVER_NAME", "Dovilo")
# Per-project signing secret from the Dovilo UI. If unset we run with
# signature checks OFF (dev only). Production must set this.
DOVILO_WEBHOOK_SECRET = os.environ.get("DOVILO_WEBHOOK_SECRET")

MODEL = "claude-sonnet-4-20250514"
MAX_AGENT_TURNS = 25
REPLAY_WINDOW_MS = 5 * 60 * 1000  # 5 minutes, per Dovilo spec
DEDUP_CAP = 1000

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is required (set it in .env)")
if not DOVILO_WEBHOOK_SECRET:
    log.warning(
        "DOVILO_WEBHOOK_SECRET is not set - signature verification DISABLED. "
        "Acceptable for local dev, not for anything else."
    )

anthropic = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)


# ---------------------------------------------------------------------------
# Dovilo MCP client
# ---------------------------------------------------------------------------
class DoviloMCPClient:
    """Long-lived stdio MCP session. Exposes tool schemas in Anthropic format."""

    def __init__(self, config_path: Path, server_name: str) -> None:
        self.config_path = config_path
        self.server_name = server_name
        self._stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None
        self.tool_schemas: list[dict[str, Any]] = []

    def _server_params(self) -> StdioServerParameters:
        cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        server_cfg = cfg["mcpServers"][self.server_name]
        return StdioServerParameters(
            command=server_cfg["command"],
            args=server_cfg.get("args") or [],
            env={**os.environ, **(server_cfg.get("env") or {})},
        )

    async def start(self) -> None:
        params = self._server_params()
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        tools_resp = await self._session.list_tools()
        self.tool_schemas = [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            }
            for t in tools_resp.tools
        ]
        log.info("[dovilo-mcp] connected - tools: %s", [t["name"] for t in self.tool_schemas])

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        assert self._session is not None, "MCP session not started"
        result = await self._session.call_tool(name, args)
        text = "\n".join(getattr(c, "text", str(c)) for c in result.content) or "(no content)"
        return {"content": text, "is_error": bool(result.isError)}


dovilo = DoviloMCPClient(DOVILO_MCP_CONFIG, DOVILO_MCP_SERVER_NAME)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt_ms(ms: Optional[int]) -> str:
    if not ms:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ms / 1000))


def _tools_for_claude() -> list[dict[str, Any]]:
    """Tools array with cache_control on the last entry to cache the whole block."""
    tools = [dict(t) for t in dovilo.tool_schemas]
    if tools:
        tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
    return tools


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------
# This is the protocol handed to the agent. Status values match Dovilo's
# lifecycle: queued → running → awaiting-user → running → done|failed|cancelled.
_SYSTEM_PROMPT = (
    "You are an agent operating inside Dovilo, a productivity app. A task has "
    "been assigned to you. You have access to MCP tools to inspect tasks, "
    "report progress, and finalize. You own the entire lifecycle — the host "
    "process will not intervene.\n\n"
    "Protocol you must follow:\n"
    "1. The user message gives you the slim task fields delivered by webhook "
    "(id, text/title, status, priority, dueAt, projectName). If you need the "
    "full body (description, context, steps, prior agent notes), call "
    "`get_task` with the task id first.\n"
    "2. Claim the task by calling `update_status` with status='running'.\n"
    "3. Work the task. Call `append_note` with concise (1-3 sentence) progress "
    "notes — that is how the user sees what you did.\n"
    "4. If the task has explicit steps (visible via get_task), call "
    "`update_step` to mark each as 'in_progress' when you start and "
    "'completed' when you finish.\n"
    "5. Finalize — always — with `update_status`, using one of:\n"
    "   • 'done'      — the task is fully addressed.\n"
    "   • 'failed'    — you cannot proceed. Append a note explaining why FIRST.\n"
    "   • 'cancelled' — the work no longer makes sense (e.g. duplicate task).\n\n"
    "Rules: be concise; don't fabricate results — if external research is "
    "needed, summarize what you would investigate and what you'd expect to "
    "find, then finalize. Never leave the task in a non-terminal status."
)


def _build_user_message(task: dict[str, Any]) -> str:
    """Render the slim webhook task into a readable prompt."""
    lines = [
        f"Task ID: {task.get('id')}",
        f"Title: {task.get('text', '')}",
        f"Current status: {task.get('status', '')}",
    ]
    if task.get("priority"):
        lines.append(f"Priority: {task['priority']}")
    if task.get("importance"):
        lines.append("Starred: yes")
    if task.get("dueAt"):
        lines.append(f"Due: {_fmt_ms(task.get('dueAt'))}")
    if task.get("projectName"):
        lines.append(f"Project: {task['projectName']}")
    lines.append("")
    lines.append(
        "Call `get_task` with this task id if you need the description, "
        "context, steps, or prior notes. Begin."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LangGraph workflow (single node — the agent owns the flow)
# ---------------------------------------------------------------------------
class WorkflowState(TypedDict):
    task: dict[str, Any]


async def agent_node(state: WorkflowState) -> dict[str, Any]:
    """Run Claude in a tool_use loop. The agent finalizes per the protocol."""
    task = state["task"]
    messages: list[dict[str, Any]] = [{"role": "user", "content": _build_user_message(task)}]
    system = [{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    tools = _tools_for_claude()

    for turn in range(MAX_AGENT_TURNS):
        resp = await anthropic.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system,
            tools=tools,
            messages=messages,
        )
        log.info(
            "[agent] turn %d stop=%s in=%d out=%d cache_r=%d cache_w=%d",
            turn + 1,
            resp.stop_reason,
            resp.usage.input_tokens,
            resp.usage.output_tokens,
            getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            break

        tool_results: list[dict[str, Any]] = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            log.info("[agent] → %s(%s)", block.name, json.dumps(block.input))
            try:
                result = await dovilo.call_tool(block.name, block.input)
                content, is_error = result["content"], result["is_error"]
            except Exception as exc:  # noqa: BLE001 - surface to the model as a tool error
                log.exception("[agent] tool %s raised", block.name)
                content, is_error = f"Tool error: {exc}", True
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
                "is_error": is_error,
            })
        messages.append({"role": "user", "content": tool_results})
    else:
        log.warning("[agent] hit MAX_AGENT_TURNS=%d for task %s", MAX_AGENT_TURNS, task.get("id"))

    return {}


def _build_graph():
    g = StateGraph(WorkflowState)
    g.add_node("agent", agent_node)
    g.set_entry_point("agent")
    g.add_edge("agent", END)
    return g.compile()


GRAPH = _build_graph()

# In-memory mirror of the latest status, updated by inbound webhooks only.
TASK_STATE: dict[str, str] = {}


async def run_workflow(task: dict[str, Any]) -> None:
    """Wake the agent for one task. Crashes are logged; no recovery."""
    try:
        await GRAPH.ainvoke({"task": task})
    except Exception:
        log.exception("[workflow] crashed for task %s", task.get("id"))


# ---------------------------------------------------------------------------
# Startup sync — fetch pending tasks once at boot
# ---------------------------------------------------------------------------
# Per Dovilo lifecycle: only 'queued' tasks are un-claimed and worth picking up.
_PENDING_STATUSES = {"queued"}


def _find_list_tasks_tool() -> Optional[str]:
    override = os.environ.get("DOVILO_LIST_TASKS_TOOL")
    if override:
        return override
    # Per the public docs, the canonical tool is `list_tasks`. Be lenient if
    # the server renames it.
    for t in dovilo.tool_schemas:
        n = t["name"].lower()
        if "task" in n and any(verb in n for verb in ("list", "query", "fetch", "search")):
            return t["name"]
    return None


def _extract_tasks(content: str) -> list[dict[str, Any]]:
    """Pull a list of task dicts out of an MCP tool result body."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [t for t in data if isinstance(t, dict) and t.get("id")]
    if isinstance(data, dict):
        for key in ("tasks", "items", "data", "results"):
            v = data.get(key)
            if isinstance(v, list):
                return [t for t in v if isinstance(t, dict) and t.get("id")]
        if data.get("id"):
            return [data]
    return []


async def _bootstrap_sync() -> None:
    """One-shot startup sync: fetch pending tasks and queue each for the agent."""
    tool_name = _find_list_tasks_tool()
    if not tool_name:
        log.info("[bootstrap] no list-tasks tool detected; skipping startup sync")
        return

    log.info("[bootstrap] fetching tasks via %s", tool_name)
    try:
        result = await dovilo.call_tool(tool_name, {})
    except Exception:
        log.exception("[bootstrap] %s call failed", tool_name)
        return

    if result["is_error"]:
        log.warning("[bootstrap] %s returned error: %s", tool_name, result["content"][:300])
        return

    tasks = _extract_tasks(result["content"])
    pending = [t for t in tasks if (t.get("status") or "").lower() in _PENDING_STATUSES]
    log.info("[bootstrap] %d task(s) returned, %d queued", len(tasks), len(pending))

    for task in pending:
        TASK_STATE[task["id"]] = task.get("status") or "queued"
        asyncio.create_task(run_workflow(task))


# ---------------------------------------------------------------------------
# Webhook security: HMAC-SHA256 over `{t}.{rawBody}` + replay window
# ---------------------------------------------------------------------------
def verify_signature(sig_header: str, raw_body: bytes) -> bool:
    """Validate X-Dovilo-Signature. Returns True if secret is unset (dev mode)."""
    if not DOVILO_WEBHOOK_SECRET:
        return True

    parts: dict[str, str] = {}
    for piece in sig_header.split(","):
        if "=" not in piece:
            continue
        k, _, v = piece.partition("=")
        parts[k.strip()] = v.strip()

    try:
        t = int(parts["t"])
        v1 = parts["v1"]
    except (KeyError, ValueError):
        return False

    now_ms = int(time.time() * 1000)
    if abs(now_ms - t) > REPLAY_WINDOW_MS:
        return False

    mac = hmac.new(
        DOVILO_WEBHOOK_SECRET.encode("utf-8"),
        f"{t}.".encode("utf-8") + raw_body,
        hashlib.sha256,
    )
    return hmac.compare_digest(mac.hexdigest(), v1)


# ---------------------------------------------------------------------------
# Idempotency: dedup by X-Dovilo-Delivery (retries reuse the same id)
# ---------------------------------------------------------------------------
_seen_deliveries: deque[str] = deque(maxlen=DEDUP_CAP)
_seen_set: set[str] = set()


def remember_delivery(delivery_id: str) -> bool:
    """Return True if this delivery is new; False if it's a duplicate."""
    if delivery_id in _seen_set:
        return False
    if len(_seen_deliveries) == DEDUP_CAP:
        evicted = _seen_deliveries[0]
        _seen_set.discard(evicted)
    _seen_deliveries.append(delivery_id)
    _seen_set.add(delivery_id)
    return True


# ---------------------------------------------------------------------------
# FastAPI webhook listener
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    await dovilo.start()
    asyncio.create_task(_bootstrap_sync())
    try:
        yield
    finally:
        await dovilo.stop()


app = FastAPI(title="Dovilo LangGraph Demo", lifespan=lifespan)


@app.post("/webhook")
async def webhook(req: Request, bg: BackgroundTasks):
    raw_body = await req.body()

    # 1) Verify signature on the raw bytes BEFORE parsing JSON.
    if not verify_signature(req.headers.get("x-dovilo-signature", ""), raw_body):
        log.warning(
            "[webhook] rejected: invalid signature (delivery=%s)",
            req.headers.get("x-dovilo-delivery"),
        )
        return Response(status_code=401, content="invalid signature")

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return Response(status_code=400, content="invalid json")

    # Prefer the headers (cheap, no body parse needed for routing).
    event_type = req.headers.get("x-dovilo-event") or payload.get("type")
    delivery_id = req.headers.get("x-dovilo-delivery") or payload.get("id") or ""

    # 2) Idempotency: same delivery id across retries — process exactly once.
    if delivery_id and not remember_delivery(delivery_id):
        log.info("[webhook] duplicate delivery %s skipped", delivery_id)
        return {"ok": True, "duplicate": True}

    if event_type == "webhook.ping":
        log.info("[webhook] ping project=%s evt=%s", payload.get("projectId"), delivery_id)
        return {"ok": True, "pong": True}

    if event_type == "task.created":
        task = payload.get("task") or {}
        if not task.get("id"):
            return {"ok": False, "reason": "missing task.id"}
        log.info(
            "[webhook] task.created id=%s text=%r priority=%s",
            task["id"], task.get("text"), task.get("priority"),
        )
        TASK_STATE[task["id"]] = task.get("status") or "queued"
        # Process async — keep the webhook response fast.
        bg.add_task(run_workflow, task)
        return JSONResponse(status_code=202, content={"ok": True, "queued": task["id"]})

    if event_type == "task.status_changed":
        task = payload.get("task") or {}
        new_status = task.get("status")
        previous = payload.get("previousStatus")
        task_id = task.get("id")
        log.info(
            "[webhook] task.status_changed id=%s %s -> %s",
            task_id, previous, new_status,
        )
        if task_id:
            TASK_STATE[task_id] = new_status or "unknown"

        # Resume signal: the task was paused for user input and the user has
        # now responded. Wake the agent again — it will re-read prior notes
        # via get_task and pick up where it left off.
        if (
            task_id
            and previous == "awaiting-user"
            and new_status not in ("done", "failed", "cancelled")
        ):
            log.info("[webhook] resuming agent for task %s (user responded)", task_id)
            bg.add_task(run_workflow, task)

        return {"ok": True, "task_id": task_id, "status": new_status, "previous": previous}

    log.warning("[webhook] unknown event: %s", event_type)
    return {"ok": False, "reason": f"unknown event: {event_type}"}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "tracked_tasks": TASK_STATE,
        "mcp_tools": [t["name"] for t in dovilo.tool_schemas],
        "signature_verification": bool(DOVILO_WEBHOOK_SECRET),
        "dedup_size": len(_seen_set),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
