import dataclasses
import itertools
from types import SimpleNamespace

import pytest

from dac.controllers import (
    DEFAULT_EPS_MULTS,
    DEFAULT_LR_MULTS,
    ActionSet,
    LinearEpsDecay,
    NoOp,
    RandomDiscrete,
    ScoreGatedLR,
    make_controller,
    random_policy_score,
    run_episode,
)
from dac.run import parse_controller_args


def test_noop_keeps_initial_values(tiny_config):
    history = run_episode(0, NoOp(), tiny_config)
    assert len(history) == tiny_config.n_blocks + 1
    assert all(result.lr == tiny_config.lr_init and result.epsilon == tiny_config.eps_init for result in history)


def test_linear_eps_decay_follows_the_line(tiny_config):
    config = dataclasses.replace(tiny_config, n_blocks=5, block_steps=60)
    controller = LinearEpsDecay(eps_start=1.0, eps_end=0.1, decay_blocks=3)
    history = run_episode(0, controller, config)
    epsilons = [result.epsilon for result in history[1:]]
    assert epsilons == pytest.approx([1.0, 0.55, 0.1, 0.1, 0.1])
    assert all(result.lr == config.lr_init for result in history)


def test_linear_eps_decay_single_block_jumps_to_the_end():
    assert LinearEpsDecay(eps_start=1.0, eps_end=0.2, decay_blocks=1).target_epsilon(1) == 0.2
    with pytest.raises(ValueError):
        LinearEpsDecay(decay_blocks=0)


def drive_with_scores(controller, config, scores: list[float]) -> list[float]:
    """Feed a controller a made-up score sequence; return the lr it sets for each block."""
    fake_run = SimpleNamespace(config=config, lr=config.lr_init, epsilon=config.eps_init, block=0, history=[])
    lrs = []
    for block, score in enumerate(scores):
        fake_run.block = block
        fake_run.history.append(SimpleNamespace(eval_score=score))
        lr_mult, eps_mult = controller(fake_run)
        assert eps_mult == 1.0
        fake_run.lr *= lr_mult
        lrs.append(fake_run.lr)
    return lrs


def test_score_gated_lr_reduces_and_restores(tiny_config):
    full = tiny_config.lr_init
    reduced = full * 0.1
    lrs = drive_with_scores(ScoreGatedLR(threshold=400, factor=0.1), tiny_config, [100, 450, 420, 300, 500])
    assert lrs == pytest.approx([full, reduced, reduced, full, reduced])


def test_score_gated_lr_sticky_stays_reduced(tiny_config):
    full = tiny_config.lr_init
    reduced = full * 0.1
    controller = ScoreGatedLR(threshold=400, factor=0.1, sticky=True)
    lrs = drive_with_scores(controller, tiny_config, [100, 450, 300, 100])
    assert lrs == pytest.approx([full, reduced, reduced, reduced])
    # A new run (block 0 again) starts unreduced.
    assert drive_with_scores(controller, tiny_config, [100])[0] == pytest.approx(full)


def test_random_discrete_is_reproducible_and_on_the_grid():
    fake_run = SimpleNamespace()  # RandomDiscrete ignores the run
    first = RandomDiscrete(seed=5)
    second = RandomDiscrete(seed=5)
    other = RandomDiscrete(seed=6)
    first_choices = [first(fake_run) for _ in range(30)]
    second_choices = [second(fake_run) for _ in range(30)]
    other_choices = [other(fake_run) for _ in range(30)]
    assert first_choices == second_choices
    assert first_choices != other_choices
    grid = set(itertools.product(DEFAULT_LR_MULTS, DEFAULT_EPS_MULTS))
    assert set(first_choices) <= grid


def test_make_controller(tiny_config):
    assert isinstance(make_controller("noop", tiny_config, 0), NoOp)
    assert isinstance(make_controller("random", tiny_config, 0), RandomDiscrete)
    assert make_controller("linear_eps", tiny_config, 0, decay_blocks=4).decay_blocks == 4
    assert make_controller("lr_gate", tiny_config, 0, sticky=True).sticky is True
    with pytest.raises(ValueError):
        make_controller("nope", tiny_config, 0)
    with pytest.raises(TypeError):
        make_controller("noop", tiny_config, 0, unexpected=1)


def test_random_policy_score_is_about_twenty(tiny_config):
    score = random_policy_score(tiny_config, episodes=100, seed=0)
    assert 15 < score < 30
    assert score == random_policy_score(tiny_config, episodes=100, seed=0)


def test_random_discrete_with_epsilon_levels(tiny_config):
    """Same draws as the default grid (one per block); the epsilon entry becomes level / current epsilon."""
    levels = (0.03, 0.1, 0.3)
    fake_run = SimpleNamespace(epsilon=0.2)
    with_levels = RandomDiscrete(seed=5, epsilon_levels=levels)
    with_multipliers = RandomDiscrete(seed=5)
    for _ in range(30):
        lr_multiplier, epsilon_multiplier = with_levels(fake_run)
        default_lr_multiplier, _ = with_multipliers(fake_run)
        assert lr_multiplier == default_lr_multiplier
        assert any(epsilon_multiplier * 0.2 == pytest.approx(level, rel=1e-12) for level in levels)

    # As built from the command line: --controller-arg epsilon_levels=0.03,0.1,0.3
    controller_args = parse_controller_args(["epsilon_levels=0.03,0.1,0.3"])
    controller = make_controller("random", tiny_config, 0, **controller_args)
    history = run_episode(0, controller, tiny_config)
    for result in history[1:]:
        assert any(result.epsilon == pytest.approx(level, rel=1e-12) for level in levels)


def test_random_discrete_with_lr_and_epsilon_levels_from_the_command_line(tiny_config):
    lr_levels, epsilon_levels = (3e-5, 1e-4, 1e-3), (0.03, 0.1, 0.3)
    controller_args = parse_controller_args(["epsilon_levels=0.03,0.1,0.3", "lr_levels=3e-5,1e-4,1e-3"])
    controller = make_controller("random", tiny_config, 0, **controller_args)
    history = run_episode(0, controller, tiny_config)
    for result in history[1:]:
        assert any(result.lr == pytest.approx(level, rel=1e-12) for level in lr_levels)
        assert any(result.epsilon == pytest.approx(level, rel=1e-12) for level in epsilon_levels)


def test_trailing_comma_gives_a_one_element_tuple(tiny_config):
    controller_args = parse_controller_args(["epsilon_levels=0.1,", "lr_levels=3e-5,1e-4,1e-3"])
    assert controller_args["epsilon_levels"] == (0.1,)
    controller = make_controller("random", tiny_config, 0, **controller_args)
    assert len(controller.action_set.pairs) == 3
    for result in run_episode(0, controller, tiny_config)[1:]:
        assert result.epsilon == pytest.approx(0.1, rel=1e-12)


def test_action_set_with_periods_repeats_each_pair_once_per_period():
    action_set = ActionSet.build((0.5, 1.0), (1.0,), lr_levels=None, period_levels=(100, 400))
    assert action_set.pairs == [(0.5, 1.0), (0.5, 1.0), (1.0, 1.0), (1.0, 1.0)]
    assert action_set.periods == [100, 400, 100, 400]
    fake_run = SimpleNamespace(lr=1e-3, epsilon=0.1)
    assert action_set.choice(1, fake_run) == (0.5, 1.0, 400)
    assert ActionSet.build((0.5, 1.0), (1.0,)).choice(1, fake_run) == (1.0, 1.0)  # no periods: two entries


def test_lr_gate_with_high_period_sets_the_period_only_while_the_lr_is_cut(tiny_config):
    controller = ScoreGatedLR(threshold=400, high_period=400)
    def run_with_last_score(score, block=1):
        return SimpleNamespace(block=block, lr=tiny_config.lr_init, config=tiny_config,
                               history=[SimpleNamespace(eval_score=score)])
    assert controller(run_with_last_score(450)) == (pytest.approx(0.1), 1.0, 400)
    assert controller(run_with_last_score(100)) == (1.0, 1.0, tiny_config.target_update_every)
    assert len(ScoreGatedLR()(run_with_last_score(450))) == 2  # default: lr only, as before


def test_lr_gate_high_period_from_the_command_line_changes_the_run(tiny_config):
    config = dataclasses.replace(tiny_config, n_blocks=4)
    controller_args = parse_controller_args(["high_period=400", "threshold=0"])  # threshold 0: cut from block 1
    assert controller_args["high_period"] == 400
    plain = run_episode(0, ScoreGatedLR(threshold=0), config)
    with_period = run_episode(0, make_controller("lr_gate", config, 0, **controller_args), config)
    assert [result.lr for result in plain] == [result.lr for result in with_period]
    # Different target copies give different q values.
    assert [result.q_mean for result in plain[1:]] != [result.q_mean for result in with_period[1:]]


def test_random_discrete_with_periods_returns_a_period_each_block():
    controller = RandomDiscrete(seed=1, lr_levels=(1e-4, 1e-3), epsilon_levels=(0.1,), period_levels=(100, 400))
    fake_run = SimpleNamespace(lr=1e-3, epsilon=0.1)
    periods = {controller(fake_run)[2] for _ in range(40)}
    assert periods == {100, 400}
