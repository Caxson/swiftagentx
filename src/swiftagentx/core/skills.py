"""
Skill-in-ReAct for SwiftAgentX v0.3.

A **Skill** is a markdown-defined workflow loaded at agent startup. The
LLM picks one to invoke from inside a ReAct iteration when the request
matches the skill's ``description`` / ``when_to_use``. Skills run
*interpreted* (the LLM follows the instructions step by step), so they
are slower than :class:`ScenarioConfig` but more flexible.

This is the complement to Scenarios — not the replacement. Scenarios
are pre-compiled execution paths picked by the LIGHT classifier; Skills
are markdown procedures invoked during open-ended ReAct reasoning when
the model decides one applies.

The frontmatter schema::

    ---
    name: refund_workflow
    description: Use when the customer has confirmed they want a refund.
    when_to_use: After the customer explicitly says they want a refund
                 AND we have their order_id.
    allowed_tools: [check_refund_eligibility, process_refund, send_confirmation]
    model_tier: heavy      # optional, defaults to heavy
    ---

    1. Check refund eligibility ...
    2. If eligible: ...
    3. If not eligible: ...

``Skill.body`` holds everything after the frontmatter block. The Agent
inserts the body into the ReAct prompt when ``invoke_skill`` is the
chosen action.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Skill:
    """A markdown-defined workflow the ReAct loop can invoke."""

    name: str
    description: str
    body: str
    when_to_use: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    model_tier: str = "heavy"
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    def to_prompt_block(self) -> str:
        """Render this skill as a block to splice into the ReAct prompt."""
        header = f"## Skill: {self.name}\n{self.description}\n"
        if self.when_to_use:
            header += f"\n**When to use:** {self.when_to_use}\n"
        if self.allowed_tools:
            header += f"\n**Allowed tools:** {', '.join(self.allowed_tools)}\n"
        return header + "\n" + self.body.strip()


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------


def parse_skill_markdown(text: str, *, source_path: Path | None = None) -> Skill:
    """
    Parse a ``Skill`` from a markdown string with YAML frontmatter.

    Tolerant: missing frontmatter is allowed (yields a Skill with name
    derived from the source filename); unknown frontmatter keys land in
    ``metadata`` instead of being errors. Required: ``name`` (or a
    source_path with a filename stem we can fall back to).
    """
    frontmatter, body = _split_frontmatter(text)
    fallback_name = source_path.stem if source_path is not None else None
    return _build_skill(frontmatter, body, source_path=source_path,
                         fallback_name=fallback_name)


# Anthropic Agent Skills frontmatter uses hyphenated keys; SwiftAgentX's own
# format uses underscores. Translate the ones we recognize before building.
_AGENT_SKILL_KEY_ALIASES = {"allowed-tools": "allowed_tools"}


def parse_agent_skill_markdown(text: str, *, source_path: Path | None = None) -> Skill:
    """
    Parse a ``Skill`` from a ``SKILL.md`` file written in the Anthropic Agent
    Skills format (YAML frontmatter with ``name`` / ``description`` and,
    optionally, hyphenated keys like ``allowed-tools``).

    Falls back to the *parent directory name* (not the ``SKILL.md`` filename
    stem) when ``name`` is missing from frontmatter, matching how Agent
    Skills packages are identified by their directory name.
    """
    frontmatter, body = _split_frontmatter(text)
    for alias, canonical in _AGENT_SKILL_KEY_ALIASES.items():
        if alias in frontmatter:
            frontmatter[canonical] = frontmatter.pop(alias)
    fallback_name = source_path.parent.name if source_path is not None else None
    return _build_skill(frontmatter, body, source_path=source_path,
                         fallback_name=fallback_name)


def _build_skill(
    frontmatter: dict[str, Any],
    body: str,
    *,
    source_path: Path | None,
    fallback_name: str | None,
) -> Skill:
    name = frontmatter.get("name")
    if name is None:
        if fallback_name is None:
            raise ValueError(
                "Skill markdown has no 'name' in frontmatter and no source_path "
                "to fall back to"
            )
        name = fallback_name

    description = frontmatter.pop("description", "") if frontmatter else ""
    when_to_use = frontmatter.pop("when_to_use", "") if frontmatter else ""
    allowed_tools = frontmatter.pop("allowed_tools", []) if frontmatter else []
    model_tier = frontmatter.pop("model_tier", "heavy") if frontmatter else "heavy"

    # 'name' was used above; remove so it doesn't leak into metadata.
    frontmatter.pop("name", None)

    if not isinstance(allowed_tools, list):
        logger.warning(
            "Skill %s: allowed_tools must be a list, got %r — coercing.",
            name, type(allowed_tools).__name__,
        )
        allowed_tools = []

    return Skill(
        name=str(name),
        description=str(description),
        when_to_use=str(when_to_use),
        body=body,
        allowed_tools=[str(t) for t in allowed_tools],
        model_tier=str(model_tier),
        metadata=dict(frontmatter),
        source_path=source_path,
    )


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a YAML frontmatter block (delimited by lines of ``---``) from body.

    If the text *clearly intended* frontmatter (starts with ``---\\n``) but is
    missing the closing ``---`` line, raise a clear ValueError. Silently
    falling back to "treat the whole thing as body" — the previous behaviour
    — meant the user's ``name: foo / description: bar`` lines were lost and
    they had no idea why (dogfood Friction #D-8).
    """
    if not text.startswith("---"):
        return {}, text
    try:
        first_break = text.index("\n", 3)
    except ValueError:
        return {}, text
    rest = text[first_break + 1:]
    try:
        end_idx = rest.index("\n---")
    except ValueError:
        raise ValueError(
            "Skill markdown starts with '---' but the frontmatter block "
            "is never closed by a matching '---' line. Add a '---' line "
            "after the last frontmatter key (e.g. after 'description:') "
            "and before the body, or remove the opening '---' if you "
            "didn't intend frontmatter."
        ) from None
    fm_block = rest[:end_idx]
    body = rest[end_idx + 4:].lstrip("\n")
    fm_data = _parse_simple_yaml(fm_block)
    return fm_data, body


def _parse_simple_yaml(block: str) -> dict[str, Any]:
    """Minimal YAML subset: scalar values and inline ``[a, b, c]`` lists.

    Avoids a hard dependency on PyYAML at the cost of supporting only the
    keys SwiftAgentX actually uses. Anything more exotic should land in
    a future commit that brings in PyYAML proper.
    """
    data: dict[str, Any] = {}
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            items = [v.strip().strip('"\'') for v in inner.split(",") if v.strip()]
            data[key] = items
        elif value.startswith(("'", '"')) and value.endswith(("'", '"')):
            data[key] = value[1:-1]
        elif value.lower() in {"true", "false"}:
            data[key] = value.lower() == "true"
        elif value.isdigit():
            data[key] = int(value)
        else:
            data[key] = value
    return data


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """
    Holds the set of known skills.

    Provides ``load_dir`` to bulk-load every ``*.md`` under a directory
    (recursive). Individual skills can also be added via ``register``.

    Bulk load failures are logged and skipped — one malformed skill file
    should never break the agent's startup.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if skill.name in self._skills:
            logger.info("Overwriting existing skill %r", skill.name)
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> bool:
        return self._skills.pop(name, None) is not None

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        return sorted(self._skills.keys())

    def load_dir(self, directory: str | Path) -> list[str]:
        """Load every ``*.md`` skill file under ``directory`` (recursive).

        Returns the list of skill names successfully registered.
        """
        d = Path(directory)
        if not d.exists():
            logger.warning("Skill dir %s does not exist", d)
            return []
        registered: list[str] = []
        for path in sorted(d.rglob("*.md")):
            try:
                skill = parse_skill_markdown(path.read_text(encoding="utf-8"),
                                             source_path=path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed skill %s: %s", path, exc)
                continue
            self.register(skill)
            registered.append(skill.name)
        return registered

    def load_agent_skills(self, directory: str | Path) -> list[str]:
        """Load Anthropic Agent Skills packages under ``directory`` (recursive).

        Each package is a subdirectory containing a ``SKILL.md`` file; any
        other files alongside it (``scripts/``, ``references/``, ``assets/``,
        etc.) are left where they are — resolve them via
        ``skill.source_path.parent`` since that points at the ``SKILL.md``
        file inside the package directory. Directories without a
        ``SKILL.md`` (including ones holding unrelated ``*.md`` files) are
        not registered as skills.

        Returns the list of skill names successfully registered.
        """
        d = Path(directory)
        if not d.exists():
            logger.warning("Agent Skills dir %s does not exist", d)
            return []
        registered: list[str] = []
        for skill_md in sorted(d.rglob("SKILL.md")):
            try:
                skill = parse_agent_skill_markdown(
                    skill_md.read_text(encoding="utf-8"), source_path=skill_md,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed Agent Skill %s: %s", skill_md, exc)
                continue
            self.register(skill)
            registered.append(skill.name)
        return registered

    def schema_for_prompt(self) -> str:
        """Render a compact "skills catalog" for the ReAct prompt.

        Used by the agent to tell the model what skills it can invoke
        via the ``invoke_skill`` action. We intentionally do NOT inline
        each skill's full body here — that's expensive on cache. We
        inline the body only when the model picks the skill.
        """
        if not self._skills:
            return ""
        lines = ["Available skills (invoke with Action: invoke_skill / Action Input: "
                 '{"skill": "<name>", "args": {...}}):']
        for skill in self._skills.values():
            line = f"- {skill.name}: {skill.description}"
            if skill.when_to_use:
                line += f" (when: {skill.when_to_use})"
            lines.append(line)
        return "\n".join(lines)
