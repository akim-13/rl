---
type: project
name: problem
description: Candidate problem (preliminary, not confirmed) — an outer RL agent tuning the hyperparameters of an inner Double DQN on CartPole — and its open risks against the spec
generated: { by: "agent/claude-fable-5-2", at: "2026-11-13T08:37:49Z" }
stale_after: 2026-11-27
---

# Problem — candidate (preliminary)

> Updated: 2026-11-13 — Akim's Claude (Fable 5.2) — first written.
> Evidence class: a group-chat message pasted by Akim on 2026-11-13 and
> described by Akim as "our preliminary idea". No group decision is recorded
> here, and no tutor has confirmed the problem as far as this brain knows.

## The idea as stated in the message

- **Problem:** tune hyperparameters so that CartPole is learned as quickly and
  as reliably as possible. An outer agent adjusts the hyperparameters of an
  inner learner.
- **Methods:** Double DQN for both the inner and the outer problem
  (preliminary). The inner method stays fixed; the investigation compares
  different RL methods for the outer problem.
- **Baseline:** hyperparameters chosen by hand. The outer agent always starts
  from them, and all results are compared against them.
- **Report plan:** the inner learner is described in the appendices and
  treated as close to a black box in the main body. Success criteria to be
  defined with the MoSCoW method.
- **Who is doing what:** Jack Ben-Dyke — Problem Definition (formal states,
  actions, transition dynamics, rewards). Adison — list candidate outer
  methods by Saturday 2026-11-14, then split the Background write-up between
  Adison, Daniel and Peter.

## Assessment against the spec

Akim's Claude (Fable 5.2), 2026-11-13, at Akim's request ("go or no go?").
This is an agent's judgment, not a ruling; nothing below was tested by
running code. Spec references are to [coursework](coursework.md).

Verdict: **not as written; go only if the three conditions below are met.**

1. **Tabular fail condition.** If the outer agent's state is just the current
   hyperparameter setting and its actions are discrete adjustments, the outer
   problem is a small discrete search — solvable by a bandit or a table. The
   inner problem does not rescue it: CartPole is routinely solved by tabular
   Q-learning over a discretised state. Condition: the outer problem is
   formulated as control *during* inner training, with a continuous state
   built from inner training statistics, so that it is a sequential decision
   problem and not a search for one good setting.
2. **Whether the outer agent can be shown to learn.** One outer episode costs
   one inner training run, and inner DQN returns vary a lot between seeds, so
   the outer reward is expensive and noisy. The pass condition requires an
   agent that clearly beats a random agent, in learning curves and on video.
   Condition: a pilot measures the wall time of one inner run and the
   seed-to-seed spread of the outer reward before the group commits.
3. **Suitability is the markers' call.** The spec tells groups to confirm the
   problem with a tutor, and an unsuitable problem is a fail. Condition: a
   tutor confirms it in a lab session.

Further points for the report, if the idea goes ahead:

- Random search with the same compute budget is the comparison a marker will
  expect; "manual baseline only" is weak on Rigour.
- The outer problem's transition dynamics *are* the inner learner, so the
  main body has to explain it enough to be understood; essential material
  moved to appendices loses marks.
- The agent-performance video needs a plan: a hyperparameter controller has
  nothing to watch unless inner CartPole behaviour is shown before and after.
- The closest literature is hyperparameter control and automated RL rather
  than meta-RL in the task-distribution sense.
