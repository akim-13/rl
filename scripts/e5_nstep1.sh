#!/usr/bin/env zsh
# E5 (2026-11-22): the best B (lr levels only, epsilon 0.1) with 1-step targets instead of 10-step,
# to measure what the n-step extension contributes. Then validate on seeds 30..59.
cd "${0:A:h}/.."
.venv/bin/python -m dac.outer_dqn --name e5_lronly_n1_s0 --seed 0 --envs 11 --rounds 36 --n-step 1 \
    --lr-levels 3e-5,1e-4,1e-3 --epsilon-levels 0.1 > results/outer_e5_lronly_n1_s0.stdout.txt 2>&1 || exit 1
.venv/bin/python -m dac.run --controller ddqn --controller-arg model=results/outer/e5_lronly_n1_s0/model.pt \
    --seeds 30:60 --workers 12 --name val_e5_lronly_n1_s0 | tail -n 1
.venv/bin/python scripts/compare.py val_lr_gate val_e3_lronly_s0 val_e5_lronly_n1_s0
