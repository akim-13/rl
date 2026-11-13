---
type: protocol
name: GUARDRAILS
description: Hard rules, never violated — read before any autonomous work
generated: { by: "agent/claude-fable-5-2", at: "2026-11-13T08:08:52Z" }
---

# GUARDRAILS — hard rules

> Changes to this file only with a principal's explicit OK (any group member
> listed in [contributors](contributors.md)), recorded in
> [decisions](decisions.md) in the same commit.

- **Never commit secrets.** `.env` values, keys, and credentials move
  person-to-person; the brain holds only pointers to where they live.
<!-- prod-freeze: not installed — no live production surface as of 2026-11-13; install when anything goes live -->
