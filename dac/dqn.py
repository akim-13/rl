"""DQN with a replay buffer and a target network (Double DQN by default).

Not CartPole-specific: works for any flat observation vector and discrete
actions, so the outer agent B reuses it.
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _relu_layers(input_size: int, sizes: tuple[int, ...]) -> list[nn.Module]:
    layers: list[nn.Module] = []
    layer_input_size = input_size
    for layer_output_size in sizes:
        layers.append(nn.Linear(layer_input_size, layer_output_size))
        layers.append(nn.ReLU())
        layer_input_size = layer_output_size
    return layers


class QNetwork(nn.Module):
    """MLP: observation -> one Q-value per action."""

    def __init__(self, observation_size: int, n_actions: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        layers = _relu_layers(observation_size, hidden)
        last_hidden_size = hidden[-1] if hidden else observation_size
        # No output activation: Q-values can be any real number
        layers.append(nn.Linear(last_hidden_size, n_actions))
        self.layers = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layers(observations)


class DuelingQNetwork(nn.Module):
    """Dueling architecture (Wang et al., 2016): Q(s, a) = V(s) + A(s, a) - mean_a A(s, a).

    Hidden layers are shared except the last, which is split into a value and
    an advantage stream. Subtracting the mean advantage makes the split into
    V and A unique. V is trained on every update whatever the action, which
    helps when the action barely matters in many states.
    """

    def __init__(self, observation_size: int, n_actions: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        if not hidden:
            raise ValueError("a dueling network needs at least one hidden layer")
        *shared_sizes, stream_size = hidden
        self.shared = nn.Sequential(*_relu_layers(observation_size, tuple(shared_sizes)))
        stream_input_size = shared_sizes[-1] if shared_sizes else observation_size
        self.value_stream = nn.Sequential(*_relu_layers(stream_input_size, (stream_size,)), nn.Linear(stream_size, 1))
        self.advantage_stream = nn.Sequential(
            *_relu_layers(stream_input_size, (stream_size,)), nn.Linear(stream_size, n_actions)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.shared(observations)
        state_values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return state_values + advantages - advantages.mean(dim=1, keepdim=True)


def make_q_network(observation_size: int, n_actions: int, hidden: tuple[int, ...], dueling: bool) -> nn.Module:
    network_class = DuelingQNetwork if dueling else QNetwork
    return network_class(observation_size, n_actions, hidden)


@dataclass
class Batch:
    """Minibatch of transitions sampled from the replay buffer, as NumPy arrays."""

    observations: np.ndarray       # (batch, observation_size) float32
    actions: np.ndarray            # (batch,) int64
    rewards: np.ndarray            # (batch,) float32
    next_observations: np.ndarray  # (batch, observation_size) float32
    terminated: np.ndarray         # (batch,) float32, 1.0 if the episode really ended (not a time-limit cut)


@dataclass
class _TensorBatch:
    """The same batch as torch tensors, sharing memory with the arrays."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor

    @classmethod
    def from_numpy(cls, batch: Batch) -> "_TensorBatch":
        return cls(
            observations=torch.from_numpy(batch.observations),
            actions=torch.from_numpy(batch.actions),
            rewards=torch.from_numpy(batch.rewards),
            next_observations=torch.from_numpy(batch.next_observations),
            terminated=torch.from_numpy(batch.terminated),
        )


class ReplayBuffer:
    """Fixed-size ring buffer of transitions; overwrites the oldest when full."""

    def __init__(self, capacity: int, observation_size: int) -> None:
        self.capacity = capacity
        self.observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.next_index = 0  # slot the next transition goes into
        self.size = 0        # slots holding real data

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        slot = self.next_index
        self.observations[slot] = observation
        self.actions[slot] = action
        self.rewards[slot] = reward
        self.next_observations[slot] = next_observation
        self.terminated[slot] = float(terminated)
        self.next_index = (slot + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        """Uniform sample with replacement. Fancy indexing returns copies, not views."""
        slots = rng.integers(0, len(self), size=batch_size)
        return Batch(
            observations=self.observations[slots],
            actions=self.actions[slots],
            rewards=self.rewards[slots],
            next_observations=self.next_observations[slots],
            terminated=self.terminated[slots],
        )

    def __len__(self) -> int:
        return self.size


def td_targets(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    next_q_online: torch.Tensor,
    next_q_target: torch.Tensor,
    gamma: float,
    double: bool,
) -> torch.Tensor:
    """One-step TD targets.

    next_q_online / next_q_target are (batch, n_actions) at the next state.
    Plain DQN picks and scores the next action with the target net, which
    overestimates; Double DQN picks with the online net and scores with the
    target net. Only a real terminal drops the bootstrap term; an episode cut
    off by the time limit still bootstraps.
    """
    if double:
        best_next_actions = next_q_online.argmax(dim=1, keepdim=True)
        next_values = next_q_target.gather(1, best_next_actions).squeeze(1)
    else:
        next_values = next_q_target.max(dim=1).values
    return rewards + gamma * (1.0 - terminated) * next_values


class DQNAgent:
    """Online and target Q-networks, replay buffer and Adam optimiser."""

    def __init__(
        self,
        observation_size: int,
        n_actions: int,
        *,
        hidden: tuple[int, ...],
        gamma: float,
        lr: float,
        buffer_size: int,
        batch_size: int,
        target_update_every: int,
        double: bool,
        grad_clip: float,
        init_seed: int,
        dueling: bool = False,
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.double = double
        self.grad_clip = grad_clip

        # The initial weights depend only on init_seed. fork_rng restores torch's
        # global RNG afterwards, so building an agent doesn't disturb other code
        # drawing from it, such as B. Nothing else in this class uses torch's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.online_net = make_q_network(observation_size, n_actions, hidden, dueling)
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.requires_grad_(False)  # only updated by copying

        # Fused Adam does the same update in one call; on a net this small the
        # per-call overhead dominates, so the step is about 3x faster.
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr, fused=True)
        self.buffer = ReplayBuffer(buffer_size, observation_size)
        self.n_updates = 0

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def set_lr(self, lr: float) -> None:
        """Change the learning rate in place; Adam keeps its moment estimates."""
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = lr

    def greedy_actions(self, observations: np.ndarray) -> np.ndarray:
        """Argmax-Q action for each row of a (rows, observation_size) array."""
        with torch.no_grad():
            q_values = self.online_net(torch.as_tensor(observations, dtype=torch.float32))
        return q_values.argmax(dim=1).numpy()

    def act(self, observation: np.ndarray, epsilon: float, coin: float, random_action: int) -> int:
        """Epsilon-greedy; random numbers come from the caller (see inner.py)."""
        if coin < epsilon:
            return int(random_action)
        single_row = observation[np.newaxis, :]
        return int(self.greedy_actions(single_row)[0])

    def store(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        self.buffer.add(observation, action, reward, next_observation, terminated)

    def update(self, rng: np.random.Generator) -> dict[str, float]:
        """One gradient step on a minibatch sampled with rng; returns diagnostics."""
        batch = _TensorBatch.from_numpy(self.buffer.sample(self.batch_size, rng))

        q_values_of_all_actions = self.online_net(batch.observations)
        q_values_of_taken_actions = q_values_of_all_actions.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        targets = self._td_targets(batch)
        loss = F.smooth_l1_loss(q_values_of_taken_actions, targets)  # Huber loss

        gradient_norm = self._gradient_step(loss)
        self._count_update_and_refresh_target()

        return _update_statistics(
            loss, targets, q_values_of_taken_actions.detach(), q_values_of_all_actions, gradient_norm
        )

    def _td_targets(self, batch: _TensorBatch) -> torch.Tensor:
        # Targets are constants: no gradient flows through them
        with torch.no_grad():
            return td_targets(
                batch.rewards,
                batch.terminated,
                next_q_online=self.online_net(batch.next_observations),
                next_q_target=self.target_net(batch.next_observations),
                gamma=self.gamma,
                double=self.double,
            )

    def _gradient_step(self, loss: torch.Tensor) -> torch.Tensor:
        """Backprop, clip and step; returns the gradient norm before clipping."""
        self.optimizer.zero_grad()
        loss.backward()
        # clip_grad_norm_ returns the norm before clipping, the more useful diagnostic
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()
        return gradient_norm

    def _count_update_and_refresh_target(self) -> None:
        self.n_updates += 1
        target_copy_is_due = self.n_updates % self.target_update_every == 0
        if target_copy_is_due:
            self.target_net.load_state_dict(self.online_net.state_dict())


def _update_statistics(
    loss: torch.Tensor,
    targets: torch.Tensor,
    q_values_of_taken_actions: torch.Tensor,
    q_values_of_all_actions: torch.Tensor,
    gradient_norm: torch.Tensor,
) -> dict[str, float]:
    td_errors = targets - q_values_of_taken_actions
    best_q_values = q_values_of_all_actions.detach().max(dim=1).values
    return {
        "loss": float(loss.detach()),
        "td_abs": float(td_errors.abs().mean()),
        "q_mean": float(best_q_values.mean()),
        "grad_norm": float(gradient_norm),
    }
