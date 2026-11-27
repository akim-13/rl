"""Inner agent A (Double DQN on CartPole) and what the outer agent B needs to drive it.

Modules:
    dqn         the learning algorithm (Q-network, replay buffer, TD targets, agent)
    inner       one training run of A, advanced block by block
    metrics     summary numbers computed from a run's evaluation scores
    controllers fixed (non-learning) ways of choosing A's hyperparameters per block
    outer_env   the training run presented as a Gymnasium environment for B
    outer_dqn   the outer agent B (a Double DQN on OuterEnv) and its greedy controller
    run         command line: run one controller over many seeds, save CSV + JSON
    report      command line: summary table and learning-curve plot from saved CSVs
    demo/       render_pilot: video of four controllers' greedy games during training
"""

import torch

# The networks are tiny (two hidden layers of 64), so extra threads cost more
# than they save. One thread per process also lets parallel runs use one core each.
torch.set_num_threads(1)
