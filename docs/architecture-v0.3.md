# SwiftAgentX v0.3 Architecture

> Status: **in progress**. This document is the construction blueprint for
> the v0.3 release cycle. Decisions captured here are binding — anything
> not listed here is out of scope for v0.3.

## Guiding principles

1. **Scenarios remain the headline abstraction.** Every v0.3 feature exists
   to make Scenarios more capable or to give ReAct the same toolbelt
   Scenarios have. Nothing in v0.3 replaces or supersedes Scenarios.
2. **Composition over reinvention.** MCP, sub-agents, skills, and hooks are
   *building blocks* a Scenario step or a ReAct iteration can call into.
   They are not parallel execution paths.
3. **Backwards compatibility is non-negotiable.** v0.2.0 users upgrading
   to v0.3 should not need to change a single line of working code unless
   they opt into a new feature.
4. **Default settings are production-sensible.** New subsystems ship with
   defaults that mirror caosen's real-world deployment experience (AI
   outbound calling at a financial services company).

## Roadmap (in implementation order)

| # | Subsystem | Status | Touches |
|---|---|---|---|
| 1 | 4-layer Memory (PersistentMemory) | 🛠 in progress | `core/memory.py`, `core/agent.py`, new `core/memory_layers.py` |
| 2 | Hook system | planned | new `core/hooks.py`, `core/agent.py`, `tools/scenario.py` |
| 3 | MCP-in-Scenario | planned | new `providers/mcp/`, `tools/scenario.py` |
| 4 | Sub-agent dispatch | planned | new `core/subagent.py`, `tools/base.py` (new tool type) |
| 5 | Skill-in-ReAct | planned | new `core/skills.py`, `core/agent.py:_react_loop` |
| 6 | Worktree-style workspace | planned | new `core/workspace.py` |
| 7 | Cache-friendly prompt ordering | planned | `core/prompt.py` |
| 8 | Lazy tool loading | planned | `tools/registry.py` |

## 1. 4-layer Memory

The current `SessionMemory` is a process-local list of `Message` objects
with naive max-age / max-count pruning. It loses everything on restart
and gives the model the most recent N messages without distinguishing
**what the model needs** (active context) from **what it needs to know
about** (background).

### Design

Four layers, in order of "freshness" and verbosity:

```
┌──────────────────────────────────────────────────────────────────────┐
│ L1  Current question                                                 │
│       The user_input being processed right now.                      │
│       Single string. Always given to the model.                      │
├──────────────────────────────────────────────────────────────────────┤
│ L2  Verbatim recent turns          (default: last 4 turns)           │
│       Full input + output pairs.                                     │
│       Most recent N turns where N is configurable.                   │
│       Always given to the model.                                     │
├──────────────────────────────────────────────────────────────────────┤
│ L3  Reference window               (default: 6 turns)                │
│       Turns older than L2 that have not yet been folded into L4.     │
│       Given to the model as "for reference" context (lower-priority).│
│       Capped at a max size; overflow forces summarization.           │
├──────────────────────────────────────────────────────────────────────┤
│ L4  Rolling summary                                                  │
│       Single text field holding the incremental summary of all       │
│       turns ever folded out of L3. Updated by re-summarizing         │
│       (L4_old + folded_turns) → L4_new.                              │
│       Always given to the model as "personal history".               │
└──────────────────────────────────────────────────────────────────────┘
```

### Summarization trigger

Re-summarization runs when **either** condition fires:

- **Cadence**: every `summarize_every_n_turns` (default 5) turns
- **Topic change** (hook): a `TopicChangeHook` runs `before_classify` on
  each new turn. It calls the LIGHT model with a prompt that asks
  *"Is this input continuing the prior conversation thread, or starting
  a new topic?"* If "new topic", `summarize()` is invoked immediately.

### Summarization algorithm

```python
async def summarize(self) -> None:
    # 1. Pick L3 turns older than L2's window.
    to_fold = self.l3_turns_outside_window()
    if not to_fold:
        return
    # 2. Build the new summary by re-prompting the LIGHT model with
    #    the *previous* summary plus the turns being folded in.
    prompt = self.prompt_manager.render("memory.summarize",
                                        prev_summary=self.l4_summary,
                                        new_turns=to_fold)
    response = await self.light_model.chat([{"role": "user", "content": prompt}])
    # 3. Atomically swap. We never keep both old summary and folded
    #    turns around — the contract is "the summary subsumes them".
    self.l4_summary = response.content
    self.l3_turns = self.l3_turns_within_window()
```

### Layered injection into the prompt

The agent's classification/ReAct/direct prompts each get all four layers,
but with different prominence:

```
<personal_history>           # L4
{l4_summary}
</personal_history>

<recent_context for_reference>     # L3
{l3_turns}
</recent_context>

<recent_dialog>              # L2 (verbatim, highest priority)
{l2_turns}
</recent_dialog>

<current_question>           # L1
{user_input}
</current_question>
```

Lower layers ground the model; the highest layer is what the model
actually responds to.

### Persistence backends

The memory itself is a pluggable interface:

```python
class MemoryBackend(ABC):
    async def load(self, session_id: str) -> MemorySnapshot: ...
    async def save(self, session_id: str, snapshot: MemorySnapshot) -> None: ...
```

v0.3 ships with `InMemoryBackend` (still process-local, but exposes the
4-layer structure) and `RedisMemoryBackend` (production-grade). The
existing `SessionMemory` becomes a thin compatibility shim that wraps
`InMemoryBackend`.

### Open question (placeholder for caosen)

- L3 hard cap before forced summarize: default 6, but should be lower for
  agents with very short SLA budgets. Make it configurable.
- Should summarization run in the foreground (blocking) or background
  (fire-and-forget after the current request)? Default: background — we
  return the response first, then summarize while the user is reading.

## 2. Hook system

### Why

Today `Agent` has hard-coded lifecycle hooks as Python methods
(`on_request_start`, `on_before_classify`, `on_before_tool_call`,
`on_before_respond`). Customizing them requires subclassing. That works
for one-off projects but doesn't scale to:

- Plugin authors who can't fork the framework
- B2B customers whose ops want to add tracing / quota / audit hooks
  declaratively
- **Semantic triggers like "topic change detected"** that aren't tied to
  a fixed lifecycle stage

A formal hook system unifies both.

### Hook taxonomy

```
Lifecycle hooks    fire at specific points in Agent.run()
    SessionStart, RequestStart, BeforeClassify, AfterClassify,
    BeforeScenarioStep, AfterScenarioStep,
    BeforeToolCall, AfterToolCall,
    BeforeReactIter, AfterReactIter,
    BeforeRespond, RequestEnd

Semantic hooks     fire when a condition evaluates true
    TopicChange, ToolFailureCascade, MaxIterationsApproaching,
    CacheHitRateLow (telemetry), CustomCondition

Scenario-step hooks  attached to specific scenario steps in tool_chain
    BeforeStep, AfterStep, OnStepFailure, ConditionalLLMCall
```

### Handler shape

A hook handler is one of:

- A Python coroutine: `async def(context, event) -> HookResult`
- A shell command path: `./hooks/check_quota.sh` — invoked with JSON on
  stdin, returns JSON on stdout. Like Claude Code's hooks.
- A `Skill` invocation (see §5) — the hook triggers a markdown-defined
  workflow.
- An LLM-prompt: `LLMHook(prompt_template="...")` — the hook is a one-shot
  LLM call whose output drives the next action.

`HookResult` carries:

```python
@dataclass(frozen=True)
class HookResult:
    action: Literal["continue", "short_circuit", "abort", "rewrite"]
    answer: str | None = None      # for short_circuit
    context_updates: dict = {}     # merged into agent context
    metadata: dict = {}
```

### Declarative configuration

```yaml
# agent_hooks.yaml
hooks:
  - event: TopicChange
    handler:
      kind: builtin
      name: trigger_memory_summarize

  - event: BeforeToolCall
    condition: tool.name == "send_email"
    handler:
      kind: shell
      command: ./check_email_quota.sh

  - event: AfterScenarioStep
    scenario: order_status
    step: courier_api
    condition: result.error is not None
    handler:
      kind: llm
      prompt: |
        The courier API failed with: {error}
        Decide if we should retry, fall back to last-known status,
        or escalate to a human. Respond with: action=retry|fallback|escalate
```

Loaded at agent boot via `Agent.load_hooks(path)`.

### Topic-change hook reference implementation

Ships as a builtin in v0.3:

```python
class TopicChangeHook(SemanticHook):
    """
    Fires before classification. Calls the LIGHT model with a prompt
    that determines whether the current user input continues the prior
    conversation thread.
    """
    event = "BeforeClassify"

    PROMPT = """
    Last 4 turns:
    {recent_turns}

    Current user input:
    {user_input}

    Is this input continuing the prior conversation or starting a new topic?
    Respond with ONLY: continuing | new_topic
    """

    async def evaluate(self, context):
        model = context.agent.light_model
        recent = context.memory.l2_turns()
        prompt = self.PROMPT.format(recent_turns=recent, user_input=context.user_input)
        response = await model.chat([{"role": "user", "content": prompt}])
        return "new_topic" in response.content.lower()

    async def on_fire(self, context):
        await context.memory.summarize()
        return HookResult(action="continue", metadata={"summary_refreshed": True})
```

## 3. MCP-in-Scenario

### Why caosen wants this

He used the same pattern in production at his financial services
employer's AI outbound calling system. The MCP ecosystem already has
50+ production-grade servers (filesystem, postgres, slack, github,
playwright, etc.). Writing Python wrapper tools for each is wasted
effort.

### Surface

```python
agent.register_mcp_server(
    name="postgres",
    transport="stdio",
    command=["/opt/mcp-postgres/server", "--db", "postgres://..."],
)

# Now the postgres MCP server's exposed tools are available as if they
# were native Python tools. Scenario tool_chain can use them:

agent.register_scenario("customer_lookup", ScenarioConfig(
    triggers=["customer info", "account details"],
    tool_chain=[
        ToolChainStep(tool="postgres.query",            # MCP tool
                      query_template="SELECT * FROM customers WHERE id=$customer_id"),
        ToolChainStep(tool="format_customer_card"),     # native Python tool
    ],
    output_type="llm_processed",
))
```

### Implementation

`providers/mcp/client.py` implements a minimal MCP JSON-RPC client over
stdio/SSE. On `register_mcp_server`:

1. Spawn the server process (stdio transport) or connect (SSE transport).
2. Call `tools/list` to discover available tools.
3. For each discovered tool, wrap it in an `MCPTool` subclass of `Tool`
   that proxies `execute()` to a JSON-RPC `tools/call`.
4. Register each as a normal tool in `ToolRegistry`, namespaced as
   `<server_name>.<tool_name>`.

The Scenario engine and the ReAct loop don't need to know an MCP tool is
different from a Python tool — the abstraction is uniform.

## 4. Sub-agent dispatch

### Why

A complex customer-service request might need three independent
look-ups: account history, recent orders, open tickets. Today the main
agent has to serialize these as ReAct iterations, polluting its context
with the intermediate results of all three lookups.

A sub-agent is a focused agent with:

- Its own `Agent` instance (own model, own tools, own context)
- A bounded mission ("look up account X's recent orders")
- A structured result schema
- Optional timeout

The main agent dispatches one or many; only the structured *results* come
back. The main agent's context stays clean.

### Surface

```python
# From inside a tool, scenario step, or ReAct iteration:
results = await agent.dispatch_subagents([
    SubAgentRequest(role="account_lookup", input=f"user_id={uid}"),
    SubAgentRequest(role="order_history",  input=f"user_id={uid}"),
    SubAgentRequest(role="open_tickets",   input=f"user_id={uid}"),
], timeout=20)
```

`SubAgent` is a registered role — defined in code or YAML, with its own
allowed tool list, prompt template, and result schema. The main agent
doesn't need to know what tools the sub-agent uses.

## 5. Skill-in-ReAct

### What Skills are (vs Scenarios)

| | Scenario | Skill |
|---|---|---|
| Definition | Python `ScenarioConfig` (tool_chain, triggers, ttl) | Markdown file with frontmatter |
| Selection | LIGHT model classifies intent → exact scenario id | LLM reads the skill's `description` and decides whether to invoke |
| Execution | Pre-compiled tool chain, no LLM thinking | Free-form, LLM follows markdown instructions step by step |
| Speed | ~200 ms | seconds (depends on instructions) |
| Use when | Frequent, predictable, latency-critical | Less frequent, requires judgment, can spend tokens |

Skills are **complementary** to Scenarios, not a replacement. They live
inside the ReAct loop: when the model is reasoning open-endedly, it can
invoke a skill to follow a structured procedure without that procedure
being baked into Python.

### Surface

```python
agent.skills_dir = Path("./skills")
# Loads all skills/*.md. Each file's frontmatter declares:
#   name, description, when_to_use, allowed_tools

# During a ReAct loop iteration, the model can emit:
#   Action: invoke_skill
#   Action Input: {"skill": "refund_workflow", "args": {...}}
```

A skill markdown file:

```markdown
---
name: refund_workflow
description: Use when the customer has confirmed they want a refund.
when_to_use: After the customer explicitly says they want a refund AND
             we have their order_id.
allowed_tools: [check_refund_eligibility, process_refund, send_confirmation]
---

1. Check refund eligibility with check_refund_eligibility(order_id=$order_id).
2. If eligible:
   - Process the refund with process_refund(order_id=$order_id, reason=$reason)
   - Send a confirmation email
3. If not eligible, explain why and offer a store credit.
```

## 6-8. Smaller subsystems

### 6. Worktree-style workspace

```python
async with agent.workspace(session_id=sid) as ws:
    await ws.write("report.pdf", data)
    file_url = await ws.upload(ws.path("report.pdf"))
# ws cleaned up
```

`workspace()` creates a temp directory scoped to the session. Used when
the agent generates files (reports, transcripts, etc.). Configurable
backend: local disk, S3, in-memory tarball.

### 7. Cache-friendly prompt ordering

Anthropic and OpenAI both reward stable prompt prefixes. The current
`PromptManager` builds prompts in an order that breaks cache locality
(memory varies per request, system prompt is stable). v0.3 enforces:

```
1. tools_section       (stable across all calls)
2. system_section      (stable across all calls)
3. l4_summary          (changes slowly, when summarize() fires)
4. l3_reference        (changes per request but tail is stable)
5. l2_recent_dialog    (changes per request)
6. l1_current_question (changes per request)
```

Saves 30-50% on cached-prompt cost for high-volume deployments.

### 8. Lazy tool loading

When `ToolRegistry.count() > config.lazy_tool_threshold` (default 20):

1. The classifier prompt no longer enumerates all tools.
2. Instead it asks the LIGHT model: *"Which categories of tools might
   this request need?"*
3. Only tools in the picked categories are revealed to the HEAVY model.

This shaves token cost when an agent has many MCP-imported tools but
typically uses 3-5 per request.

---

## Out of scope for v0.3

Listed so we don't drift:

- Permission policy (default-deny / require-confirmation per tool) — caosen
  said "default should be configured properly", not a runtime concern.
- Slash commands / chat UI affordances — SwiftAgentX is API-first.
- Single binary CLI — SwiftAgentX is a library, not a CLI.
- Output styles (Explanatory / Learning / etc.) — use prompt templates.
- Status line / dashboard UI — admin API endpoint exists, UI is downstream.
- System-reminder-style mid-conversation interjections — request lifetime
  is too short in production agents to need this.
