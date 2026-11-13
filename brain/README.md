---
type: protocol
name: README
description: How this brain works — organisation + the maintenance protocol
generated: { by: "agent/claude-fable-5-2", at: "2026-11-13T08:08:52Z" }
---

# brain/ — shared memory for rl (humans + agents)

> Updated: 2026-11-13 — brain initialized from the shared brain spec (Akim's Claude (Fable 5.2))

This folder is the **single source of truth** for project knowledge that isn't
derivable from the code: current status, decisions and why, open questions,
todos. Plain markdown committed to the repo, so it travels with a `git clone`
to any machine or agent with **zero install** — no MCP server, no database.

## How it's organised

- **`STATUS.md`** — the front door. Only what is true now: current focus,
  done / in progress / blocked, next action. Read this first, every session.
- **`index.md`** — the index. One line per file here.
- **`log.md`** — dated session/update history, newest first.
- **`decisions.md`** — append-only decision log.
- **Fact files** — atomic knowledge, one concept per file, frontmatter per the
  format below.
- **`GUARDRAILS.md`** — hard rules, never violated.
- **`TODO.md`** — shared task list.
- **`contributors.md`** — who works here (the principal roster).

<!-- brain:core v4 -->
## File format (OKF v0.2)

This brain is an [Open Knowledge Format v0.2](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
bundle: plain markdown, one concept per file, YAML frontmatter. Every file except
the reserved `index.md` / `log.md` carries frontmatter with at least `type:`:

```yaml
---
type: project            # project | reference | decision | feedback | dashboard | protocol | lesson
name: dev-setup          # stable slug — the file's historical link target
description: One-line summary used to judge relevance
updated: 2026-08-04      # legacy freshness field; kept until the file's next real edit
---
```

On any meaningful edit, record freshness with the OKF trust fields instead of
bumping `updated:` — `generated: { by: <actor>, at: <ISO datetime> }` (actor is
`human:<id>`, `agent/<model>`, or `process:<id>`), plus optionally
`verified: [{ by, at }]` (who confirmed the content against reality — distinct
from who wrote it) and `stale_after: YYYY-MM-DD` (absolute date after which a
volatile fact is suspect). Cross-link with standard markdown links relative to
this folder, e.g. `[infrastructure](infrastructure.md)`. Dashboards
(`STATUS.md`, `TODO.md`) carry the same frontmatter plus a human-facing
`> Updated: YYYY-MM-DD — <who/why>` header.

## Maintenance protocol

**Invariant: zero rot, anywhere, ever** (`zero-rot`). No file, row, or claim in
this brain may sit stale-but-plausible. Every rule below serves this; if you
touch the brain and notice rot — an outdated claim, a dead link, index drift, an
"in progress" that finished weeks ago — fixing it is part of the current
session, not a someday-task.

1. **STATUS.md is the front door** (`status-front-door`). Every session starts
   by reading this README + `STATUS.md`, then only the files relevant to the
   task.
2. **index.md stays in sync** (`index-in-sync`). Add or remove a file → update
   its line in `index.md` the same session.
3. **Dated, always** (`dated-always`). Freshness fields in frontmatter for fact
   files, a `> Updated: YYYY-MM-DD — <who/why>` header for dashboards.
   **Absolute ISO dates only** — never "today" / "next week".
4. **Single source of truth** (`single-source`). Each fact lives in exactly one
   file; elsewhere link to it, never copy.
5. **Newest-dated wins** (`newest-wins`). On conflict, the newer-dated entry is
   authoritative; fix or remove the stale one in the same session.
6. **decisions.md is append-only** (`decisions-append-only`). Never edit a past
   decision — supersede it with a new dated entry.
7. **End-of-session write-back** (`writeback-covers-negative-decisions`).
   Before finishing a work session: rewrite `STATUS.md` to the now-true state
   (+ its date), append the session narrative to `log.md`, log decisions,
   tick/prune `TODO.md`. "Log decisions" includes decisions reached in
   conversation that touched no code — the decision NOT to build something
   included — with the rationale and the alternatives rejected. Chat
   transcripts are not recoverable; `decisions.md` is the only durable WHY.
   Write-back also checks: did this session edit text inside a managed block
   (`<!-- brain:* -->` markers), or improve a rule in a way other projects
   would want? If so, flag it for backport to the shared module source **now**
   — an improvement stranded in one project's block is invisible everywhere
   else, and the divergence will stall future syncs.
8. **Narratives point, they don't retell** (`narrative-is-pointer`). When a
   decisions entry or a spec amendment captures a session's substance, the
   `log.md` entry and any `STATUS.md` bullet are pointers — 1–3 lines per
   item: what changed · where the substance lives · what's owed. Full log
   narrative is reserved for sessions whose story has no better home
   (exploration, incident context, multi-item batches whose arc no single
   decision holds). This is `single-source` applied to prose: every extra
   framing of the same fact is a future drift site, and the redundancy tax
   concentrates exactly where write-back is most routine.
9. **Verify volatile facts before relying on them** (`verify-volatile`).
   Anything older than a few weeks: re-confirm before acting.
10. **Keep it lean** (`keep-lean`). Prune done items; archive superseded
   material rather than letting it rot in place.
11. **An oversized file must be dealt with, never ridden** (`size-invariant`).
    A brain file over **1,500 lines** (archives excluded; a project may set a
    different cap only by a recorded decision) is a defect the noticing
    session must act on: (a) shrink it honestly — rotate closed/historical
    content to an archive folder verbatim, links re-based, a pointer/index
    line left behind; (b) restructure it, if the content is live but the file
    conflates concerns; or (c) declare a dated frontmatter exemption
    (`oversize: accepted <date> — <why>; revisit when <condition>`), reserved
    for deliberate single-file designs or restructures needing a principal's
    ruling — an exemption is a flag, not a burial: surface it in the session
    report. Where the repo has a test suite, give the cap mechanical teeth (a
    test that fails on undeclared oversize — reference implementation:
    demux `tests/test_brain_size.py`); a size rule that lives only in prose
    goes unnoticed until a principal trips over the file.
11. **Evidence class travels with the claim** (`evidence-discipline`). Every
    claim about external state written anywhere in this repo names **how** and
    **when** it was observed: live API call · live page fetch · seen in a UI by
    a named human · cached/synced file (quote that file's own date) ·
    inference. Phrase the claim as the observation itself — "`data/reviews.json`
    (synced 15 Jul) says 28", never "the API returns 28" unless someone called
    the API that day. Corollaries: (a) a local clone of any repo that receives
    bot/CI commits is **stale by default** — pull or read the remote before
    quoting its files; (b) never write a mechanism nobody observed ("the cron
    ran") — write the evidence ("bot commit `58684f4`, 09:06 UTC"); (c) when
    re-reading a stored claim, the recorded evidence class decides whether it
    needs re-verification before being repeated or acted on.
12. **Anchors are stable identifiers** (`stable-anchors`). References in brain
    files point at stable identifiers — paths, rule/spec IDs, PR numbers,
    env-var names — never line numbers, test counts, or dates-as-status (they
    rot fastest and check worst).

## What belongs here vs. NOT here

A brain keeps everything that should live in the repo. In a shared repo, that
is only what's relevant to all collaborators; in a solo repo, most project
knowledge — generalized so work can resume from another machine.

- **Here:** anything a teammate or another machine needs — launch steps, infra
  ownership, caveats/gotchas, decisions, strategy, open questions, todos.
  Write it **machine-agnostically**: no absolute home-dir paths, no personal
  ssh aliases, no "the file I made earlier" references.
- **NOT here:** secrets (`.env` values, keys, creds — those move
  person-to-person; the brain holds only a pointer to where they live),
  machine-specific state (local paths, personal tool config, one person's
  dev-machine workflow), or anything the code / git history / project docs
  already record.

## Cross-store rule (for agents with their own memory)

This `brain/` is the source of truth for project knowledge. A machine-local
agent memory (e.g. Claude's `~/.claude/.../memory/`) must hold **only
machine-specific state and a pointer here** — never a copy of project facts.
That keeps the stores from diverging and keeps every collaborator's agent
fully served by a plain `git pull`.

*`.manifest.json` in this folder is tooling metadata: it records which shared
brain modules/versions generated this brain and its declared local
customizations. Don't index it in `index.md`; leave it alone except when
deliberately syncing.*
<!-- /brain:core -->

<!-- brain:collaboration v1 -->
## Ownership & authority

- **Decisions record the decider** (`ruling-provenance`). Every decisions.md
  entry ends with the decider(s) in parens. Distinguish roles when they
  differ: the person who ruled is the decider ("Ratified by Akim");
  advisors/agents are contributors ("(Akim + Claude)" means
  Akim ruled, Claude participated). An agent-only entry ("(Claude)")
  is a judgment call, not a ruling — any principal may overturn it freely.
- **All principals have equal authority** (every group member listed in
  [contributors](contributors.md)). No one's rulings take precedence; each
  is free to change project rules and specs. There is no domain split.
- **Supersession must cite and answer.** An entry that changes an existing
  ruling names the superseded entry and states why its rationale no longer
  holds. No silent contradictions — citing forces reading.
- **Cross-principal changes get flagged, not blocked.** When a superseded
  entry's deciders include another principal and they're not party to the
  new one (joint entries count), the change lands immediately, but the writing
  agent adds a TODO addressed to them: "entry X supersedes entry Y — needs
  <name>'s ack." Ack = intentional change, tick it; objection = contradiction,
  reconcile in conversation and log the outcome as a new entry. Never block on
  input; record and resolve retrospectively.
- **Session artifacts name their author.** Log entries, STATUS `> Updated:`
  headers, and any session records say which human/agent wrote them (e.g.
  "Akim's Claude", "<name>") — with several people's agents
  writing here, provenance of observations matters as much as dates.
<!-- /brain:collaboration -->

## Project rules (local)

- **Decisions use stable `D-###` IDs.** Each entry in
  [decisions](decisions.md) gets the next ID; a superseding entry cites the
  superseded entry by its D-ID.
- **Coursework facts come from the spec, not from memory.** Unit code,
  deadline, deliverables and marking are recorded in a `coursework.md` fact
  file once the spec is in the repo, each with the spec file it was read from
  and the date it was read. Until that file exists, none of those facts are
  known to this brain (see [STATUS](STATUS.md)).
