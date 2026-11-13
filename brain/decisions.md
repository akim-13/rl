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

## D-004 — 2026-11-13 — The repo is hosted on GitHub as the private repository `akim-13/rl`
Rationale: the group needs one shared remote and every member has a GitHub
account. Akim asked for the repo to be created and for the four other members
to be invited; no other host was discussed. Private, because the spec requires the
group to work independently of other groups (`coursework/spec.md`, Academic
Integrity). The name `rl` (after the project folder) and the private
visibility were picked by the agent and are open to change. Who has access:
[contributors](contributors.md).
(Akim; name and visibility by Akim's Claude (Fable 5.2))

## D-005 — 2026-11-13 — A working copy of the report is a separate git clone in `report/`, ignored by this repo
Rationale: the report's remote is Overleaf, which asks each person for their
own token. As a submodule it would make every clone of the code repo depend
on Overleaf access; a plain clone in an ignored folder keeps the two
histories apart and leaves the code repo cloneable by anyone. Alternatives
rejected: a submodule (above), and a clone outside the repo folder (an agent
working in this repo would not find it). The same arrangement was already in
use in another of Akim's projects (seen on Akim's machine, 2026-11-13).
Details: [report](report.md).
(Akim's Claude (Fable 5.2) — a judgment call, not a ruling)
