#!/usr/bin/env zsh
# E6 (2026-11-22): B chooses A's lr (3e-5/1e-4/1e-3) and target-copy period (100/400), epsilon fixed at 0.1;
# 6 actions, 10-step targets, 36 rounds, B seeds 0 and 1 in parallel. Then validate both on seeds 30..59
# against lr_gate and against the matching hand rule (lr_gate with high_period=400).
cd "${0:A:h}/.."
common=(--envs 11 --rounds 36 --n-step 10 --lr-levels 3e-5,1e-4,1e-3 --epsilon-levels 0.1 --period-levels 100,400)
for seed in 0 1; do
  .venv/bin/python -m dac.outer_dqn --name e6_period_s$seed --seed $seed $common > results/outer_e6_period_s$seed.stdout.txt 2>&1 &
done
wait
for seed in 0 1; do
  .venv/bin/python -m dac.run --controller ddqn --controller-arg model=results/outer/e6_period_s$seed/model.pt \
      --seeds 30:60 --workers 12 --name val_e6_period_s$seed | tail -n 1
done
.venv/bin/python scripts/compare.py val_lr_gate val_lr_gate_hp400 val_e6_period_s0 val_e6_period_s1 val_e6_period_s0+val_e6_period_s1
.venv/bin/python scripts/compare.py val_lr_gate_hp400 val_e6_period_s0 val_e6_period_s1 val_e6_period_s0+val_e6_period_s1
