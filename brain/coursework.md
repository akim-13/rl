---
type: reference
name: coursework
description: CM32032 Assessment 1 (function-approximated RL project) — where the spec lives, dates, deliverables, limits, penalties, fail conditions, marking
generated: { by: "agent/claude-fable-5-2", at: "2026-11-13T08:32:01Z" }
stale_after: 2026-12-04
---

# Coursework — CM32032 Reinforcement Learning, Assessment 1

> Updated: 2026-11-13 — Akim's Claude (Fable 5.2) — first written.
> Evidence class: read from `coursework/CM32032-CW1-spec-26-27.pdf` (all ten
> pages) and the files in `coursework/latex-template/` on 2026-11-13. Akim
> downloaded both that day; the PDF's own creation date is 2026-10-30. Whether
> a newer version has been published since was not checked.
> This file is a digest for quick checks. The wording that counts is the
> spec's — read [the full text](../coursework/spec.md) before any work on a
> deliverable.

## Where things are

| What | Path | Notes |
|---|---|---|
| Spec, original | `coursework/CM32032-CW1-spec-26-27.pdf` | Authoritative. |
| Spec, full text | [coursework/spec.md](../coursework/spec.md) | Verbatim transcription for reading and search; loaded into agent sessions through `AGENTS.md`. |
| LaTeX template, as provided | `coursework/latex-template/` | Pristine copy — never edit it; copy it to start the report. The original zip sits next to it. |
| Group contribution form | not in the repo | The spec refers to a "provided group contribution form document"; it has not been added (see [TODO](TODO.md)). |

## Key facts

- **Unit:** CM32032 Reinforcement Learning. Assessment 1, "Function-Approximated
  RL Project". The spec names Joshua B. Evans and Sophia Jones as its authors.
- **Due:** Thursday 2027-01-07, 20:00, on Moodle. One group member uploads
  everything.
- **Weight:** 50% of the unit mark.
- **Group:** four to six students. The group formation deadline (2026-11-06,
  20:00) had passed when this file was written. Who is in this group:
  [contributors](contributors.md).
- **Marking is not anonymous.** Individual marks are the group mark adjusted
  by the group contribution form.
- **GenAI:** Open Lane — use is allowed and must be declared honestly in the
  report (section 8 below). The group stays responsible for everything
  submitted, and each member must be able to explain the overall approach and
  their own contribution.
- **Task:** apply function-approximated RL to a problem of the group's choice
  and investigate how well the methods solve it. The chosen problem is not
  recorded in this brain yet.

## Fail conditions (each one alone gives a failing grade)

- The problem can be solved effectively with tabular methods. The state/action
  space must be too large for that; the spec says to check the choice with a
  tutor in a lab session.
- No own implementation of at least one function-approximated RL method. A
  ready-made implementation from a library or repository earns no marks and
  may appear only as a baseline, clearly labelled as such.
- No agent demonstrates noticeable learning (the minimum for a pass: clearly
  better than a random agent, shown in the video and in learning curves).

## Deliverables (all five mandatory; −10 marks for each one missing)

| Deliverable | Format | Limit |
|---|---|---|
| Project report | `.pdf`, not zipped | 7 pages of main body |
| Video presentation | video file (`.mp4`, `.mov`, `.mkv`) | 4 minutes, natural speaking tempo, every member speaks |
| Agent performance video (before and after training) | video file, separate from the presentation | 2 minutes; may be sped up if still clear |
| Source code | one `.zip` | Must include a short README explaining how to run it and reproduce the results |
| Group contribution form | `.pdf`, not zipped | — |

Files too large for Moodle: upload a `.txt` of links instead; the links must
stay publicly accessible for at least one year after the deadline.

## Report rules

Sections, in this order (not following the structure costs 5 marks):

1. Problem Definition — states, actions, transition dynamics, reward function;
   why tabular methods cannot solve it.
2. Background — candidate methods, strengths and weaknesses, with literature.
3. Method — what was implemented, how it works in the group's own words, why.
4. Results — how quickly and how well the agents learn; informative baselines
   and comparisons.
5. Discussion — critical evaluation, limitations.
6. Future Work
7. Personal Experience
8. Generative AI Use Declaration — which tools, for what, to what extent, and
   where relevant how different members used them. Outside the page limit.
9. References — outside the page limit; cite everything discussed, including
   lecture material.
10. Appendices — outside the page limit; must be referenced from the main body.

- **Seven pages** for sections 1–7, including figures, tables and captions.
  Nothing beyond page seven of the main body is read.
- Written with the provided LaTeX template, which must not be modified. How
  agents apply this (Akim's Claude's reading): fill in the title, authors and
  section bodies of `rl_project_template.tex`; leave `rl_project.sty` and the
  page-layout settings untouched.
- Figures and any text inside them must be comfortably legible, or marks are
  lost.
- From `rl_project_formatting_guide.tex`: no title page and no table of
  contents; the report starts with Problem Definition directly under the
  title and authors. Appendices not referenced in the main body are not read,
  and moving essential material into appendices to save space loses marks.
  Captions go below figures and above tables. Plots should stay legible in
  greyscale. Overleaf is the recommended way to co-write.

## Penalties

- −10 marks per missing deliverable.
- −5 marks each for: exceeding a page or time limit; not using the template;
  not following the report structure; presentation not at normal speaking
  tempo; not every member speaking in the presentation.
- Penalties add up without a cap, but cannot turn a passing mark into a fail.

## Marking

Four criteria, assessed across report, presentation, agent video and code:
**Quality**, **Understanding**, **Rigour**, **Presentation**. Final agent
performance is only one part of Quality, judged relative to the difficulty of
the problem and the time and compute available.

The grade descriptors for above 50%, 60% and 70% are in the "Grade
Descriptors" section of [the full text](../coursework/spec.md); they are too
detailed to digest here without losing what distinguishes the bands. The spec
expects the mean grade to fall between 50% and 70% and says grades above 70%
will normally be rare.

## Third-party code

Libraries for generic functionality (PyTorch, TensorFlow) and existing
environments (Gymnasium) are allowed, and an existing environment is strongly
recommended — there are no marks for building one. Any third-party code used
in any part of the implementation must be acknowledged in the source-code
README: a reference, how it was used, and what was changed.
