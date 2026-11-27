#!/usr/bin/env bash
# Tabular baselines for B (decisions D-011, D-012), one B seed per arm.
# Arm 1, tab_lronly_s0 (score x block, 30 states), was started separately with:
#   .venv/bin/python -m dac.outer_tabular --name tab_lronly_s0 --seed 0
# This script: bin ranges for arm 2 from fixed-settings runs on training seeds, trains arm 2
# (all 8 features), waits for arm 1, then evaluates both on VAL (30-59) and TEST (0-29).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

[ -f results/trainseeds_noop.csv ] || $PY -m dac.run --controller noop --seeds 1000:1030 --workers 6 --name trainseeds_noop
$PY -m dac.outer_tabular --name tab8_lronly_s0 --seed 0 --state all_features --bin-range-from trainseeds_noop \
    > results/outer_tab8_lronly_s0.stdout.txt 2>&1
until [ -f results/outer/tab_lronly_s0/train_meta.json ]; do sleep 60; done

for arm in tab tab8; do
    $PY -m dac.run --controller tabular --controller-arg model=results/outer/${arm}_lronly_s0/q_table.json \
        --seeds 30:60 --workers 12 --name val_${arm}_lronly_s0
    $PY -m dac.run --controller tabular --controller-arg model=results/outer/${arm}_lronly_s0/q_table.json \
        --seeds 0:30 --workers 12 --name p2_${arm}_lronly_s0
done
