# CM32032 coursework code: adjusting a DQN's hyperparameters while it trains

An inner agent A (Double DQN) learns Gymnasium's `CartPole-v1`. One training
run of A is split into blocks of environment steps. Between blocks, a
controller may multiply A's learning rate and exploration rate (epsilon). This
repository contains A, the block-by-block training run, fixed (non-learning)
controllers used as baselines, `OuterEnv` (the training run presented as a
Gymnasium environment, so that an outer agent B can act as the controller),
the outer agent B (the same Double DQN, learning to act as the controller), and
scripts to run many seeds and summarise them.

## Setup

Python 3.11, CPU only.

```
python3.11 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
```

## Tests

```
.venv/bin/python -m pytest -q                  # everything, including one full-size training run
.venv/bin/python -m pytest -q -m "not slow"    # skip the full-size run
```

## Running a batch and making the report

```
.venv/bin/python -m dac.run --controller noop --seeds 0:30 --workers 10 --name baseline
.venv/bin/python -m dac.run --controller random --seeds 0:30 --workers 10 --name random
.venv/bin/python -m dac.report results/baseline.csv results/random.csv --out results/plots/baseline_vs_random.png
```

- `--controller`: `noop` (never change anything), `random` (random multipliers
  from the grid), `linear_eps` (epsilon decays linearly), `lr_gate` (learning
  rate cut once the score is high).
- `--seeds A:B` runs seeds A to B−1; `--workers` runs that many seeds at once
  (one CPU core each).
- `--set key=value` overrides any `InnerConfig` field, e.g. `--set block_steps=500`
  or `--set hidden=128,128`. `--set dueling=true` gives A a dueling network
  (Wang et al., 2016: separate value and advantage heads) instead of a plain one.
- `--controller-arg key=value` passes an argument to the controller, e.g.
  `--controller-arg decay_blocks=15` or `--controller-arg sticky=true`.
  `lr_gate` also takes `high_period=400`: A's target-copy period is 400 while
  the learning rate is cut, and A's `target_update_every` otherwise.
- Results go to `results/<name>.csv` (one row per seed and block) and
  `results/<name>.json` (config, controller, seeds, timings, library versions).
  Existing files are only replaced with `--overwrite`.

`dac.report` prints, for each CSV, the mean area under the learning curve (AUC)
with a 95% bootstrap confidence interval, its spread across seeds, the mean
final score, how many runs reached the solved threshold and how fast, and the
time per run; for several CSVs it also prints the seed-by-seed AUC difference
against the first one. It saves a plot of mean evaluation score against
environment steps with 95% bootstrap bands.

A run is fully determined by its seed, its config and the controller's choices:
running the same command twice on the same machine and library versions gives
the same numbers (apart from the timing columns).

## Files

- `dac/__init__.py`: limits PyTorch to one thread per process (tiny networks; many runs share the CPU).
- `dac/dqn.py`: Q-network, replay buffer, TD targets (Double DQN or plain DQN), the DQN agent.
- `dac/inner.py`: `InnerConfig` (all settings), `BlockResult` (what is recorded per block), `InnerRun` (one training run, advanced block by block, with greedy evaluation after every block).
- `dac/metrics.py`: AUC, outer return, steps to threshold, blocks above threshold, bootstrap confidence interval.
- `dac/controllers.py`: the fixed controllers, `ActionSet` (an action grid and how its pairs become multipliers, shared with `OuterEnv` and B), `make_controller`, `run_episode` (one full run with a controller), the random-policy reference score.
- `dac/outer_env.py`: `OuterEnv`, the training run as a Gymnasium environment (one episode = one run, one step = one block), and B's provisional observation features.
- `dac/run.py`: command line to run one controller over many seeds in parallel and save CSV + JSON.
- `dac/report.py`: command line to print the summary table and plot learning curves from saved CSVs.
- `dac/outer_dqn.py`: the outer agent B — n-step transitions, training loop over parallel `OuterEnv`s, `LearnedController` (B used greedily as a controller), command line.
- `dac/demo/render_pilot.py`: renders the video of A's CartPole behaviour under several controllers.
- `tests/`: pytest tests (buffer, TD targets, determinism, multipliers, evaluation isolation, truncation handling, `OuterEnv`, controllers, metrics, command-line parsing, the outer agent, one full-size learning check). `tests/test_golden.py` reruns the command lines and compares their output bit for bit with reference files in `tests/golden/`, written by `tests/make_golden.py` before the 2026-11-21 refactor; never regenerate them to make a test pass.

## Training the outer agent B

```
.venv/bin/python -m dac.outer_dqn --name ddqn_s0 --seed 0 --envs 11 --rounds 36
.venv/bin/python -m dac.run --controller ddqn --controller-arg model=results/outer/ddqn_s0/model.pt \
    --seeds 0:30 --workers 10 --name p2_ddqn_s0
```

- `--envs` training runs of A are stepped in lockstep in parallel processes;
  `--rounds` episodes each, so `envs × rounds` runs of A in total (396 above,
  about an hour on a 6-core laptop). B's training seeds are drawn from 1,000
  upwards, never from the test seeds 0–29.
- `--n-step N` (default 1): B learns from N-step transitions. Each stored
  transition sums the next N rewards, discounted by γ per block, and B's
  target bootstraps from the observation N blocks later (discounted by γ^N);
  transitions starting within N blocks of the end of a run are cut short
  there and do not bootstrap. With 1, B uses ordinary one-step targets.
- `--epsilon-levels 0.03,0.1,0.3` (default: off): the epsilon entry of B's
  actions is an absolute epsilon for the next block instead of a multiplier
  (the learning-rate entry stays a multiplier), so repeated choices cannot
  drive epsilon to an extreme. The list must contain A's `eps_init`, so that
  B has a "change nothing" action. The same option exists for the random
  baseline: `--controller random --controller-arg epsilon_levels=0.03,0.1,0.3`.
- `--lr-levels 3e-5,1e-4,1e-3` (default: off): the same for the learning-rate
  entry, so B can move A's learning rate tenfold in one block. The list must
  contain A's `lr_init`. Random baseline: `--controller-arg lr_levels=...`.
  A single level needs a trailing comma (`epsilon_levels=0.1,`); B's
  `--epsilon-levels 0.1` takes it without one.
- `--period-levels 100,400` (default: off): each (lr, epsilon) action is
  repeated once per period, and the action also sets A's target-copy period
  (`target_update_every`) for the next block. The list must contain A's
  `target_update_every`. Random baseline: `--controller-arg period_levels=100,400`.
  The period chosen is not written to the results CSV.
- `--dueling` (default: off): B uses a dueling network (a DQN variant) instead
  of a plain one.
- Writes to `results/outer/<name>/`: `model.pt` (B's final network),
  `checkpoints/round_<NN>.pt` (B's network after each round, same format;
  `NN` counts from 00 like the `round` column of the log), `train_log.csv`
  (one row per training episode: inner seed, epsilon, AUC, how often each
  action was taken), `replay_buffer.npz` (every transition B stored, as
  arrays `observations`, `actions`, `rewards`, `next_observations`,
  `terminated`) and `train_meta.json`.
- The printout after each round includes B's highest Q-value at one fixed
  observation (the first block-0 observation of round 0) next to the largest
  discounted return possible; a Q-value above that is an overestimate.
- The trained B is then evaluated greedily like any controller (`--controller
  ddqn`); a model trained with `--epsilon-levels`, `--lr-levels`,
  `--period-levels` or `--dueling` uses them there too. `model=a.pt,b.pt`
  averages the Q-values of several trained B networks (an ensemble).

## What we implemented and what is third-party

Written for this project: the Double DQN agent (network, replay buffer, TD
targets, update step), the block-by-block training run and its evaluation, the
outer environment, the controllers, the outer agent B (which reuses the same
Double DQN code), and the run and report scripts.

Third-party libraries, used for generic functionality only: PyTorch provides
tensors, automatic differentiation and the Adam optimiser; Gymnasium provides
the CartPole simulation and the environment interface; NumPy provides arrays
and random number generators; Matplotlib draws the plots; pytest runs the
tests. No third-party reinforcement learning implementation is used, and no
code was copied or adapted from other repositories.
