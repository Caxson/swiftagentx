"""
Tests for v0.3 Skill-in-ReAct.

Covers:

- Frontmatter parsing (full, missing fields, missing frontmatter entirely,
  malformed, list parsing)
- ``SkillRegistry`` register / unregister / list / get
- ``load_dir`` walks recursively, skips malformed files, returns registered names
- ``Skill.to_prompt_block`` formatting
- ``SkillRegistry.schema_for_prompt`` catalog output
- ``Agent.invoke_skill`` end-to-end with DummyModelClient
"""

from __future__ import annotations

from pathlib import Path

import pytest

from swiftagentx import Agent, DummyModelClient, SwiftAgentConfig
from swiftagentx.core.skills import (
    Skill,
    SkillRegistry,
    parse_agent_skill_markdown,
    parse_skill_markdown,
)

FIXTURE_AGENT_SKILLS_DIR = Path(__file__).parent / "fixtures" / "agent_skills"

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parse_full_frontmatter() -> None:
    md = (
        "---\n"
        "name: refund_workflow\n"
        "description: Use when refund is confirmed\n"
        "when_to_use: After confirmation AND order_id known\n"
        "allowed_tools: [check_eligibility, process_refund]\n"
        "model_tier: heavy\n"
        "---\n"
        "\n"
        "1. Check eligibility\n"
        "2. Process refund\n"
    )
    skill = parse_skill_markdown(md)
    assert skill.name == "refund_workflow"
    assert skill.description == "Use when refund is confirmed"
    assert skill.allowed_tools == ["check_eligibility", "process_refund"]
    assert skill.model_tier == "heavy"
    assert "1. Check eligibility" in skill.body


def test_parse_no_frontmatter_uses_filename(tmp_path: Path) -> None:
    file = tmp_path / "my_skill.md"
    file.write_text("Just a body, no frontmatter.")
    skill = parse_skill_markdown(file.read_text(), source_path=file)
    assert skill.name == "my_skill"
    assert skill.description == ""
    assert skill.body == "Just a body, no frontmatter."


def test_parse_partial_frontmatter() -> None:
    md = "---\nname: minimal\n---\nbody\n"
    skill = parse_skill_markdown(md)
    assert skill.name == "minimal"
    assert skill.description == ""
    assert skill.allowed_tools == []


def test_parse_missing_name_without_path_raises() -> None:
    with pytest.raises(ValueError):
        parse_skill_markdown("---\ndescription: x\n---\nbody")


def test_parse_unknown_keys_go_to_metadata() -> None:
    md = "---\nname: x\nteam: ops\nrequest_id_format: uuid4\n---\nbody"
    skill = parse_skill_markdown(md)
    assert skill.metadata.get("team") == "ops"
    assert skill.metadata.get("request_id_format") == "uuid4"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_register_and_list() -> None:
    r = SkillRegistry()
    r.register(Skill(name="a", description="A", body="bodyA"))
    r.register(Skill(name="b", description="B", body="bodyB"))
    assert r.list_skills() == ["a", "b"]
    assert r.get("a") is not None
    assert r.get("missing") is None


def test_registry_unregister() -> None:
    r = SkillRegistry()
    r.register(Skill(name="x", description="", body=""))
    assert r.unregister("x") is True
    assert r.unregister("x") is False


def test_load_dir_walks_recursively(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("---\nname: a\n---\nbody A")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.md").write_text("---\nname: b\n---\nbody B")

    r = SkillRegistry()
    loaded = r.load_dir(tmp_path)
    assert sorted(loaded) == ["a", "b"]


def test_load_dir_skips_malformed(tmp_path: Path) -> None:
    """load_dir survives unreadable / unparseable files without aborting the batch."""
    (tmp_path / "good.md").write_text("---\nname: good\n---\nbody")

    # Make a file with bytes that aren't valid UTF-8.
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe binary garbage \x00")

    r = SkillRegistry()
    loaded = r.load_dir(tmp_path)
    assert loaded == ["good"]


def test_load_dir_filename_fallback_when_no_name(tmp_path: Path) -> None:
    """Files with frontmatter missing 'name' should still load via filename stem."""
    (tmp_path / "fallback.md").write_text("---\ndescription: no name\n---\nbody")
    r = SkillRegistry()
    loaded = r.load_dir(tmp_path)
    assert loaded == ["fallback"]


def test_load_dir_missing_dir_returns_empty(tmp_path: Path) -> None:
    r = SkillRegistry()
    assert r.load_dir(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_to_prompt_block_includes_metadata() -> None:
    skill = Skill(
        name="refund",
        description="Refund flow",
        when_to_use="after confirmation",
        body="step 1\nstep 2",
        allowed_tools=["t1", "t2"],
    )
    block = skill.to_prompt_block()
    assert "## Skill: refund" in block
    assert "Refund flow" in block
    assert "after confirmation" in block
    assert "t1, t2" in block
    assert "step 1" in block


def test_schema_for_prompt_lists_all_skills() -> None:
    r = SkillRegistry()
    r.register(Skill(name="a", description="A desc", body=""))
    r.register(Skill(name="b", description="B desc",
                     when_to_use="when ...", body=""))
    schema = r.schema_for_prompt()
    assert "- a: A desc" in schema
    assert "- b: B desc" in schema
    assert "(when: when ...)" in schema


def test_schema_for_prompt_empty_returns_empty_string() -> None:
    assert SkillRegistry().schema_for_prompt() == ""


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_invoke_skill_runs_through_model() -> None:
    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False))
    agent.register_skill(Skill(
        name="hello",
        description="Say hello in a fancy way",
        body="Greet the user enthusiastically.",
    ))

    output = await agent.invoke_skill("hello", args={"to": "world"},
                                      context_input="from a happy user")
    assert isinstance(output, str)
    assert output  # DummyModelClient returns non-empty


@pytest.mark.asyncio
async def test_agent_invoke_unknown_skill_returns_marker() -> None:
    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False))
    output = await agent.invoke_skill("nope")
    assert "not found" in output


@pytest.mark.asyncio
async def test_agent_load_skills_picks_up_directory(tmp_path: Path) -> None:
    (tmp_path / "s.md").write_text("---\nname: s\ndescription: D\n---\nbody")

    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False))
    loaded = agent.load_skills(tmp_path)
    assert loaded == ["s"]
    assert agent.skills.get("s") is not None


# ---------------------------------------------------------------------------
# D8: Anthropic Agent Skills (SKILL.md) format loader
# ---------------------------------------------------------------------------


def test_parse_agent_skill_maps_hyphenated_allowed_tools() -> None:
    md = (
        "---\n"
        "name: pdf-tools\n"
        "description: Use for PDF extraction\n"
        "allowed-tools: [read_file, write_file]\n"
        "license: Apache-2.0\n"
        "---\n"
        "\n"
        "1. Extract text.\n"
    )
    skill = parse_agent_skill_markdown(md)
    assert skill.name == "pdf-tools"
    assert skill.allowed_tools == ["read_file", "write_file"]
    assert skill.metadata.get("license") == "Apache-2.0"


def test_parse_agent_skill_falls_back_to_parent_dir_name(tmp_path: Path) -> None:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("---\ndescription: no name field\n---\nbody")

    skill = parse_agent_skill_markdown(skill_md.read_text(), source_path=skill_md)
    assert skill.name == "my-skill"
    assert skill.source_path == skill_md


def test_registry_load_agent_skills_finds_nested_skill_md(tmp_path: Path) -> None:
    (tmp_path / "pkg-a").mkdir()
    (tmp_path / "pkg-a" / "SKILL.md").write_text("---\nname: pkg-a\n---\nbody A")
    (tmp_path / "pkg-a" / "references").mkdir()
    (tmp_path / "pkg-a" / "references" / "doc.md").write_text("not a skill")
    (tmp_path / "pkg-b").mkdir()
    (tmp_path / "pkg-b" / "SKILL.md").write_text("---\nname: pkg-b\n---\nbody B")

    r = SkillRegistry()
    loaded = r.load_agent_skills(tmp_path)
    assert sorted(loaded) == ["pkg-a", "pkg-b"]
    # The reference doc is not itself registered as a skill.
    assert r.get("doc") is None


def test_registry_load_agent_skills_skips_dirs_without_skill_md(tmp_path: Path) -> None:
    (tmp_path / "not-a-skill").mkdir()
    (tmp_path / "not-a-skill" / "notes.md").write_text("random markdown")

    r = SkillRegistry()
    assert r.load_agent_skills(tmp_path) == []


def test_registry_load_agent_skills_missing_dir_returns_empty(tmp_path: Path) -> None:
    r = SkillRegistry()
    assert r.load_agent_skills(tmp_path / "nope") == []


def test_registry_load_agent_skills_source_path_resolves_resource_dir(tmp_path: Path) -> None:
    skill_dir = tmp_path / "pkg-a"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: pkg-a\n---\nbody")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "run.py").write_text("# resource")

    r = SkillRegistry()
    r.load_agent_skills(tmp_path)
    skill = r.get("pkg-a")
    assert skill is not None
    resource_dir = skill.source_path.parent
    assert (resource_dir / "scripts" / "run.py").exists()


def test_load_agent_skills_real_fixture_sample() -> None:
    """Loads the real pdf-tools SKILL.md fixture (Anthropic Agent Skills format)."""
    r = SkillRegistry()
    loaded = r.load_agent_skills(FIXTURE_AGENT_SKILLS_DIR)
    assert loaded == ["pdf-tools"]

    skill = r.get("pdf-tools")
    assert skill is not None
    assert skill.allowed_tools == ["read_file", "write_file"]
    assert skill.metadata.get("license") == "Apache-2.0"
    resource_dir = skill.source_path.parent
    assert (resource_dir / "scripts" / "extract.py").exists()
    assert (resource_dir / "references" / "split_notes.md").exists()


@pytest.mark.asyncio
async def test_agent_load_agent_skills_and_invoke_end_to_end() -> None:
    """Real SKILL.md sample loaded and triggered through the ReAct invoke path."""
    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False))
    loaded = agent.load_agent_skills(FIXTURE_AGENT_SKILLS_DIR)
    assert loaded == ["pdf-tools"]

    output = await agent.invoke_skill(
        "pdf-tools", args={"input": "report.pdf"}, context_input="user wants text extracted",
    )
    assert isinstance(output, str)
    assert output
