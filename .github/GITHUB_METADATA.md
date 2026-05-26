# GitHub Repository Metadata — Recommendations

> **Do not commit this file.** It's a checklist for the maintainer to apply
> in the GitHub Settings UI and via `gh` CLI.

## 1. Repository description (Settings → General → Description)

> **Production Agent framework with sub-second latency. Tiered execution
> (cache / scenario / ReAct), dual-model routing, three-level cache. A
> sharper alternative to LangChain for predictable workloads.**

Keep it under 350 chars (GitHub limit). The one above is 230.

## 2. Topics (Settings → General → Topics)

Aim for 12-15 tags. GitHub uses these for the "Topics" search index, and
they meaningfully affect discoverability. Recommended set:

```
agent-framework
ai-agent
llm
python
react-agent
openai
async
fastapi
flask
rag
knowledge-base
streaming
sse
production-ready
low-latency
```

Apply via CLI:

```bash
gh repo edit Caxson/swiftagentx --add-topic agent-framework,ai-agent,llm,python,react-agent,openai,async,fastapi,flask,rag,knowledge-base,streaming,sse,production-ready,low-latency
```

## 3. Social preview image (Settings → General → Social preview)

GitHub displays a 1280×640 image when the repo URL is shared on
X/LinkedIn/Slack. Recommended content:

  - Top half: **"SwiftAgentX"** in a heavy sans-serif, with the tagline
    *"Production Agent framework. Sub-second by design."* underneath.
  - Bottom half: a stripped-down version of the tiered execution table from
    the README (cache 0 ms → scenario 200 ms → ReAct 4-10 s).
  - Background: dark slate (`#0f172a`) with a single accent color
    (`#f59e0b` works well).
  - Logo: optional — even a single character mark is fine.

Tools that work well for this without a designer:

  - https://www.canva.com/ (search "GitHub social preview" template)
  - https://socialify.git.ci/ — generate one from the repo metadata in
    30 seconds, edit colors and pattern. Then download and upload.

PNG, 1280×640, under 1 MB.

## 4. Repository settings checklist

| Setting | Recommended | Why |
|---------|-------------|-----|
| Default branch | `main` | already correct |
| Restrict push to `main` (branch protection) | ✅ | require PR review even for solo dev — looks professional |
| Require status checks (CI) to pass before merging | ✅ | once `.github/workflows/ci.yml` is in place |
| Allow squash merging | ✅ | clean history |
| Disable merge commits | optional | personal preference |
| Allow auto-merge | ✅ | useful with Dependabot later |
| Automatically delete head branches | ✅ | hygiene |

## 5. Files audit

| File | Status | Action |
|------|--------|--------|
| `LICENSE` | ✅ present | Apache-2.0 |
| `README.md` | ✅ rewritten in v0.2 prep | — |
| `CHANGELOG.md` | ✅ added in v0.2 prep | — |
| `CONTRIBUTING.md` | ✅ added in v0.2 prep | — |
| `CODE_OF_CONDUCT.md` | ✅ added in v0.2 prep | — |
| `SECURITY.md` | ❌ missing | low priority — add when you have ≥100 stars |
| `.github/workflows/ci.yml` | ✅ added in v0.2 prep | — |
| `.github/ISSUE_TEMPLATE/` | ❌ missing | low priority; consider after first external issue |
| `.github/PULL_REQUEST_TEMPLATE.md` | ❌ missing | low priority |
| `.github/FUNDING.yml` | ❌ missing | only meaningful if you're seeking sponsorship |
| `.github/dependabot.yml` | ❌ missing | nice-to-have; auto-bumps pydantic, httpx, etc. |

## 6. PyPI metadata

PyPI surfaces the `pyproject.toml` URLs and classifiers. After v0.2.0 ships,
verify on https://pypi.org/project/swiftagentx/ that:

- [ ] "Project links" shows GitHub / Issues / Changelog (not the broken
      `swiftagent/swiftagent` URL)
- [ ] "Author" shows `Caxson` (not "SwiftAgent Team")
- [ ] "Development Status :: 4 - Beta" classifier is present
- [ ] "Requires Python: >=3.10" is shown
- [ ] Long description (README) renders correctly — sometimes badges break

PyPI uses the README in `description-content-type = "text/markdown"`. Since
that's not set explicitly in `pyproject.toml`, hatchling infers it from the
`.md` extension. If badges don't render, set
`description-content-type = "text/markdown"` explicitly.

## 7. Activity signals

GitHub's "Trending" and "Suggested for you" algorithms reward:

- **Stars over time** — concentrated bursts beat one-shot floods.
- **Recent commits** — 7-day decay; nothing on day 8.
- **Active issues with maintainer replies** — shows the project is alive.
- **Releases with notes** — v0.2.0 with a changelog gives the algorithm
  something to surface.

The two highest-leverage moves right now:

1. **Ship v0.2.0 with the rewritten README, benchmarks, and CI badge.**
   This converts every drive-by visitor better.
2. **Post a write-up.** A "How I built a sub-second Agent framework"
   technical post (HN, dev.to, X) drives the first 50-200 stars. With the
   v0.2.0 numbers in the README, you can lead with the benchmark.
