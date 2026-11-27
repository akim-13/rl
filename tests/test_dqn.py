import numpy as np
import pytest
import torch

from dac.dqn import DQNAgent, DuelingQNetwork, QNetwork, ReplayBuffer, td_targets

OBSERVATION_SIZE = 4


def make_agent(**overrides) -> DQNAgent:
    settings = dict(
        hidden=(16, 16),
        gamma=0.99,
        lr=1e-3,
        buffer_size=100,
        batch_size=8,
        target_update_every=3,
        double=True,
        grad_clip=10.0,
        init_seed=0,
    )
    settings.update(overrides)
    return DQNAgent(OBSERVATION_SIZE, 2, **settings)


def transition(number: int, observation_size: int = OBSERVATION_SIZE) -> tuple:
    """Transition `number`; its fields encode the number so that sampled rows can be traced."""
    observation = np.full(observation_size, number, dtype=np.float32)
    action = number % 2
    reward = float(number)
    next_observation = observation + 0.5
    terminated = number % 3 == 0
    return observation, action, reward, next_observation, terminated


def fill(buffer: ReplayBuffer, count: int, observation_size: int = OBSERVATION_SIZE) -> None:
    for number in range(count):
        buffer.add(*transition(number, observation_size))


def fill_agent(agent: DQNAgent, count: int) -> None:
    for number in range(count):
        agent.store(*transition(number))


def all_parameters_equal(first: torch.nn.Module, second: torch.nn.Module) -> bool:
    return all(torch.equal(first_parameter, second_parameter)
               for first_parameter, second_parameter in zip(first.parameters(), second.parameters()))


def test_qnetwork_shape():
    network = QNetwork(4, 3, (8, 8))
    assert network(torch.zeros(5, 4)).shape == (5, 3)


def test_dueling_network_shape_and_mean_zero_advantage():
    network = DuelingQNetwork(4, 3, (8, 8))
    observations = torch.randn(5, 4)
    q_values = network(observations)
    assert q_values.shape == (5, 3)
    # The mean over actions of Q equals V, because the advantages are centred.
    features = network.shared(observations)
    assert torch.allclose(q_values.mean(dim=1, keepdim=True), network.value_stream(features), atol=1e-6)


def test_dueling_agent_learns_a_fixed_target():
    """A dueling agent fits constant per-action rewards on terminal transitions."""
    agent = DQNAgent(2, 2, hidden=(16, 16), gamma=0.9, lr=1e-2, buffer_size=100, batch_size=32,
                     target_update_every=10, double=True, grad_clip=10.0, init_seed=0, dueling=True)
    for step in range(100):
        action = step % 2
        agent.buffer.add(np.zeros(2, np.float32), action, float(action), np.zeros(2, np.float32), True)
    rng = np.random.default_rng(0)
    for _ in range(300):
        agent.update(rng)
    with torch.no_grad():
        q_values = agent.online_net(torch.zeros(1, 2))[0]
    assert q_values[0].item() == pytest.approx(0.0, abs=0.05)
    assert q_values[1].item() == pytest.approx(1.0, abs=0.05)


def test_buffer_wraps_around_at_capacity():
    buffer = ReplayBuffer(capacity=3, observation_size=2)
    fill(buffer, 5, observation_size=2)
    assert len(buffer) == 3
    # Transitions 0 and 1 were overwritten by 3 and 4; slot 2 still holds 2.
    assert buffer.observations[:, 0].tolist() == [3.0, 4.0, 2.0]
    assert buffer.rewards.tolist() == [3.0, 4.0, 2.0]
    assert buffer.next_index == 2


def test_buffer_sample_shapes_and_dtypes():
    buffer = ReplayBuffer(capacity=10, observation_size=OBSERVATION_SIZE)
    fill(buffer, 6)
    batch = buffer.sample(7, np.random.default_rng(0))  # more than stored: with replacement
    assert batch.observations.shape == (7, 4) and batch.observations.dtype == np.float32
    assert batch.next_observations.shape == (7, 4) and batch.next_observations.dtype == np.float32
    assert batch.actions.shape == (7,) and batch.actions.dtype == np.int64
    assert batch.rewards.shape == (7,) and batch.rewards.dtype == np.float32
    assert batch.terminated.shape == (7,) and batch.terminated.dtype == np.float32
    # Each sampled row is one consistent stored transition, never an empty slot.
    assert np.all(batch.observations[:, 0] == batch.rewards)
    assert np.all(batch.next_observations == batch.observations + 0.5)
    assert np.all(batch.rewards < 6)


def test_buffer_sample_is_reproducible():
    buffer = ReplayBuffer(capacity=10, observation_size=OBSERVATION_SIZE)
    fill(buffer, 10)
    first = buffer.sample(5, np.random.default_rng(1))
    second = buffer.sample(5, np.random.default_rng(1))
    assert np.array_equal(first.rewards, second.rewards)


def test_td_targets_double_vs_plain():
    # gamma = 0.5 keeps every number exactly representable, so == is safe.
    rewards = torch.tensor([1.0, 0.5, 2.0])
    terminated = torch.tensor([0.0, 0.0, 1.0])
    next_q_online = torch.tensor([[1.0, 3.0], [2.0, 0.0], [9.0, 0.0]])  # online picks 1, 0, 0
    next_q_target = torch.tensor([[5.0, 2.0], [4.0, 7.0], [8.0, 6.0]])

    double = td_targets(rewards, terminated, next_q_online, next_q_target, gamma=0.5, double=True)
    plain = td_targets(rewards, terminated, next_q_online, next_q_target, gamma=0.5, double=False)

    # Double: target net scores the online net's choice: 1 + 0.5*2, 0.5 + 0.5*4.
    assert torch.equal(double, torch.tensor([2.0, 2.5, 2.0]))
    # Plain: max of the target net: 1 + 0.5*5, 0.5 + 0.5*7.
    assert torch.equal(plain, torch.tensor([3.5, 4.0, 2.0]))
    assert not torch.equal(double, plain)


def test_td_targets_terminal_is_just_the_reward():
    rewards = torch.tensor([1.0, -3.0])
    terminated = torch.ones(2)
    next_q = torch.tensor([[100.0, 50.0], [-20.0, 40.0]])
    for double in (True, False):
        targets = td_targets(rewards, terminated, next_q, next_q, gamma=0.99, double=double)
        assert torch.equal(targets, rewards)


def test_same_init_seed_gives_same_weights_and_target_starts_as_copy():
    first, same_seed, other_seed = make_agent(init_seed=3), make_agent(init_seed=3), make_agent(init_seed=4)
    assert all_parameters_equal(first.online_net, same_seed.online_net)
    assert all_parameters_equal(first.online_net, first.target_net)
    assert not all_parameters_equal(first.online_net, other_seed.online_net)


def test_set_lr_changes_every_param_group():
    agent = make_agent()
    agent.set_lr(5e-4)
    assert agent.lr == 5e-4
    assert all(parameter_group["lr"] == 5e-4 for parameter_group in agent.optimizer.param_groups)


def test_act_uses_coin_and_epsilon():
    agent = make_agent()
    observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    greedy = int(agent.greedy_actions(observation[None, :])[0])
    other = 1 - greedy
    assert agent.act(observation, epsilon=0.5, coin=0.2, random_action=other) == other  # explore
    assert agent.act(observation, epsilon=0.5, coin=0.7, random_action=other) == greedy  # exploit
    assert agent.act(observation, epsilon=0.0, coin=0.0, random_action=other) == greedy


def test_greedy_actions_batch():
    agent = make_agent()
    actions = agent.greedy_actions(np.zeros((5, OBSERVATION_SIZE), dtype=np.float32))
    assert actions.shape == (5,)
    assert set(actions.tolist()) <= {0, 1}


def test_update_returns_finite_stats_and_copies_target_on_schedule():
    agent = make_agent(target_update_every=3)
    fill_agent(agent, 50)
    rng = np.random.default_rng(0)
    for update_number in range(1, 7):
        statistics = agent.update(rng)
        assert set(statistics) == {"loss", "td_abs", "q_mean", "grad_norm"}
        assert all(isinstance(value, float) and np.isfinite(value) for value in statistics.values())
        # Hard copy after updates 3 and 6 only.
        copy_was_due = update_number % 3 == 0
        assert all_parameters_equal(agent.online_net, agent.target_net) == copy_was_due


def test_construction_leaves_torch_global_rng_untouched_but_seeds_the_weights():
    torch.manual_seed(123)
    state = torch.get_rng_state()
    first = make_agent()
    assert torch.equal(state, torch.get_rng_state())

    # The weights still depend only on init_seed, whatever the global state is.
    torch.manual_seed(999)
    second = make_agent()
    assert all_parameters_equal(first.online_net, second.online_net)


def test_update_does_not_touch_torch_global_rng():
    agent = make_agent()
    fill_agent(agent, 50)
    state = torch.get_rng_state()
    agent.update(np.random.default_rng(0))
    agent.greedy_actions(np.zeros((2, OBSERVATION_SIZE), dtype=np.float32))
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("double", [True, False])
def test_plain_and_double_both_learn_a_one_state_problem(double):
    """Every transition is terminal with reward 1 for action 0 and 0 for action 1."""
    agent = make_agent(double=double, lr=1e-2, batch_size=32)
    observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    for number in range(64):
        action = number % 2
        agent.store(observation, action, 1.0 - action, observation, True)
    rng = np.random.default_rng(0)
    for _ in range(300):
        agent.update(rng)
    with torch.no_grad():
        q_values = agent.online_net(torch.from_numpy(observation[None, :]))[0]
    assert q_values[0].item() == pytest.approx(1.0, abs=0.05)
    assert q_values[1].item() == pytest.approx(0.0, abs=0.05)
