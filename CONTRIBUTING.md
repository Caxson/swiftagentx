# Contributing to SwiftAgentX

Thanks for considering a contribution! SwiftAgentX is small enough that this
guide is short on purpose — if anything below is unclear, please open an
issue and we'll fix it.

## Quick start

```bash
git clone https://github.com/Caxson/swiftagentx.git
cd swiftagentx
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,openai,flask,fastapi,benchmark]"
pytest -q
```

The full test suite takes under a second on most machines.

## Development loop

```bash
# Format and lint
ruff check --fix src/ tests/
ruff format src/ tests/

# Type check
mypy src/

# Tests with coverage
pytest --cov=src --cov-report=term-missing

# Benchmarks (mock LLM mode — no API key required)
python benchmarks/run_benchmarks.py --mode mock
```

## Pull request checklist

Before opening a PR:

- [ ] All tests pass locally (`pytest -q`)
- [ ] `ruff check src/ tests/` is clean
- [ ] `mypy src/` introduces no new errors
- [ ] New behavior is covered by a test
- [ ] CHANGELOG.md `## [Unreleased]` section updated if user-facing
- [ ] Public API changes are reflected in `docs/`

For changes that touch the execution path (`core/agent.py`, `core/cache.py`,
`tools/scenario.py`, `core/router.py`), please also run the benchmark suite
and paste the before/after numbers into the PR description. This is the only
way we can keep the "sub-second response" claim honest.

## Commit style

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add async streaming to OpenAI provider
fix: handle None timestamp in SessionMemory cleanup
docs: rewrite README positioning section
chore: bump version to 0.2.0
perf: skip ReAct for cache-hit scenarios
test: add concurrency tests for CacheManager
```

`feat:` / `fix:` / `perf:` show up in release notes. `chore:` / `docs:` /
`test:` / `refactor:` / `ci:` don't.

## Architecture notes for reviewers

When reviewing changes that touch the request path, the question that matters
most is **does this preserve the latency invariants?**

| Path | Invariant |
|------|-----------|
| Cache hit | Zero LLM calls. No async I/O beyond the dict lookup. |
| Scenario | One LIGHT-model classification call, then a synchronous tool chain. |
| ReAct | Heavy model, bounded by `max_iterations`. |
| Direct | One HEAVY-model call. |

A regression that turns a cache-hit path into one extra LLM call is a P0 bug
— even if all tests pass.

## Code of Conduct

By participating in this project you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md). Be kind. Engage with ideas, not people.

## Licensing

SwiftAgentX is licensed under Apache-2.0. By contributing, you agree that
your contributions will be licensed under the same terms.
