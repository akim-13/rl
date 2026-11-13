---
type: reference
name: report
description: Where the project report is written (Overleaf), how agents reach it over git, and where a working copy lives
generated: { by: "agent/claude-fable-5-2", at: "2026-11-13T16:48:17Z" }
---

# Report — where it lives and how to reach it

The report is written in an Overleaf project:
<https://www.overleaf.com/project/6af5dea469bacf594ec6ca77>. Akim gave this
link by message on 2026-11-13 and said agents are to reach the report through
Overleaf's git integration.

What the report must satisfy (template, page limit, section order) is in
[coursework](coursework.md).

## Git access

- **Remote:** `https://git.overleaf.com/6af5dea469bacf594ec6ca77` — the
  project ID from the link above on Overleaf's git host. Its only branch is
  `main`.
- **Login:** user name `git`; the password is an Overleaf Git authentication
  token, which each person generates in their own Overleaf Account Settings.
- A token belongs to an Overleaf account, not to a project: one token works
  for every project that account owns or collaborates on. An account can hold
  at most 10 tokens and each expires after one year. So each member needs
  their own token, and none is needed per project. (Overleaf's documentation
  page "Git integration authentication tokens" on docs.overleaf.com, fetched
  2026-11-13. That page does not say whether git integration needs a paid
  plan, or whose.)
- The token is a credential. It stays on its owner's machine and is never
  committed ([GUARDRAILS](GUARDRAILS.md)).

Evidence: on 2026-11-13 at 16:48Z the remote was fetched from Akim's machine
with Akim's token, so git integration works for this project. Whether it
works with another member's token has not been tried.

## Working copy

A member's working copy is a plain git clone in `report/` at the root of this
repo. This repo ignores that folder, so the report's history and the token
stay out of it ([D-005](decisions.md)).

```
git clone https://git.overleaf.com/6af5dea469bacf594ec6ca77 report
```

To avoid typing the token on every fetch, point the clone's own credential
helper at a file inside its `.git` folder:
`git -C report config credential.helper "store --file=<absolute path>/report/.git/overleaf-credentials"`.

- **The clone is stale by default.** The others edit in Overleaf, so pull in
  `report/` before reading or quoting the report.
- Pushing to this remote changes the document every member sees in Overleaf.

## What the project held

Read from the clone at Overleaf commit `494acc3` (dated
2026-11-13T16:36Z, the only commit), on 2026-11-13:

- Four files, the same names as the provided template in
  `coursework/latex-template/`. `rl_project.sty`,
  `rl_project_formatting_guide.tex` and `some_figure.pdf` are byte-identical
  to the provided copies.
- The report itself is being written in `rl_project_template.tex`, which had
  a draft of the Problem Definition section and some added packages.
