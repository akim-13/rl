#!/usr/bin/env zsh
# Test look #3 (2026-11-22): the hand rule lr_gate + high_period=400 (period 400 chosen on validation
# seeds 30..59) evaluated once on test seeds 0..29, compared with lr_gate on the same seeds.
cd "${0:A:h}/.."
.venv/bin/python -m dac.run --controller lr_gate --controller-arg high_period=400 \
    --seeds 0:30 --workers 12 --name p2_lr_gate_hp400 | tail -n 1
.venv/bin/python scripts/compare.py p2_lr_gate p2_lr_gate_hp400
