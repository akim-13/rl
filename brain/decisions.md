---
type: decision
name: decisions
description: Append-only decision log — never edit a past entry, supersede with a new dated one
generated: { by: "agent/claude-fable-5-2", at: "2026-11-13T08:08:52Z" }
---

# Decisions (append-only, newest last)

Entries carry stable `D-###` IDs; a superseding entry cites the D-ID it replaces.

## D-001 — 2026-11-13 — Brain initialized from the shared brain spec
Rationale: single canonical protocol; improvements propagate via module sync.
(Akim's Claude (Fable 5.2); Akim asked for the setup)

## D-002 — 2026-11-13 — Collaboration rules installed; all group members have equal authority
Rationale: this is a group coursework repo, so several people's agents will
write here and authorship and ownership rules are needed from the start. Akim
chose equal authority for all members over the alternative of Akim owning the
repo rules. These rules are new to this repo, so every other principal must
ack them (see [TODO](TODO.md)). The roster in
[contributors](contributors.md) is incomplete as of this entry.
(Akim; recorded by Akim's Claude (Fable 5.2))

## D-003 — 2026-11-13 — Agent entry file is AGENTS.md; CLAUDE.md is a one-line stub
Rationale: group members may use agents other than Claude, and a cross-agent
entry file costs nothing. `AGENTS.md` at the repo root holds the entry text;
`CLAUDE.md` only imports it. Alternative rejected: a Claude-only `CLAUDE.md`.
(Akim; recorded by Akim's Claude (Fable 5.2))
