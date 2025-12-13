#!/usr/bin/env python3
"""
Dueling DQN Agent with Prioritized Experience Replay and Curriculum Learning
for Balatro Poker Environment.

Architecture:
- Dueling Network: Separates V(s) and A(s,a) streams for better value estimation
- Prioritized Experience Replay: Focuses learning on important transitions
- Curriculum Learning: Trains across increasing target scores for generalization

This agent is designed to outperform tabular Q-learning by:
1. Using neural networks to generalize across similar states
2. Learning from the most informative experiences
3. Training on progressively harder difficulties
"""

import sys
import os
import argparse
import yaml
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
from collections import deque
import random

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")

from balatro_env import BalatroBatchedSimEnv
from q_learning_utils import plot_training_curves, compare_to_baseline, create_model_filename
from random_strategy_agent import evaluate_random_strategy_baseline
from agent_visualizer import AgentVisualizer
from strategy_action_encoder import StrategyActionEncoder


# =============================================================================
# STATE PREPROCESSOR
# =============================================================================

class StatePreprocessor:
    """
    Converts observation dictionary to fixed-size feature tensor for neural network.

    Features (37 total):
    - plays_left / 4 (1)
    - discards_left / 3 (1)
    - chips / target_score (1)
    - chips_to_target / target_score (1)
    - best_hand_score / target_score (1)
    - best_hand_type one-hot (9)
    - flush_potential, straight_potential (2)
    - has_pair, has_trips, has_two_pair, has_full_house,
      has_four_of_kind, has_straight, has_flush, has_straight_flush (8)
    - card_ranks normalized (8)
    - card_suits normalized (8)
    """

    STATE_DIM = 40

    def __init__(self, target_score: int):
        self.target_score = target_score

    def process(self, obs: Dict) -> np.ndarray:
        """Convert observation to feature vector."""
        features = []

        # Resource features (normalized)
        features.append(obs['plays_left'] / 4.0)
        features.append(obs['discards_left'] / 3.0)

        # Progress features (normalized by target)
        chips = obs['chips'][0] if hasattr(obs['chips'], '__len__') else obs['chips']
        chips_to_target = obs['chips_to_target'][0] if hasattr(obs['chips_to_target'], '__len__') else obs['chips_to_target']
        best_hand_score = obs['best_hand_score'][0] if hasattr(obs['best_hand_score'], '__len__') else obs['best_hand_score']

        features.append(min(chips / self.target_score, 2.0))
        features.append(min(chips_to_target / self.target_score, 2.0))
        features.append(min(best_hand_score / self.target_score, 2.0))

        # Best hand type one-hot (9 categories)
        hand_type = obs['best_hand_type']
        hand_one_hot = [0.0] * 9
        if 0 <= hand_type < 9:
            hand_one_hot[hand_type] = 1.0
        features.extend(hand_one_hot)

        # Draw potential (binary)
        features.append(float(obs.get('flush_potential', 0)))
        features.append(float(obs.get('straight_potential', 0)))

        # Hand classification flags (binary)
        features.append(float(obs.get('has_pair', 0)))
        features.append(float(obs.get('has_trips', 0)))
        features.append(float(obs.get('has_two_pair', 0)))
        features.append(float(obs.get('has_full_house', 0)))
        features.append(float(obs.get('has_four_of_kind', 0)))
        features.append(float(obs.get('has_straight', 0)))
        features.append(float(obs.get('has_flush', 0)))
        features.append(float(obs.get('has_straight_flush', 0)))

        # Card ranks (normalized 0-1)
        card_ranks = obs['card_ranks']
        for rank in card_ranks:
            features.append(rank / 12.0)

        # Card suits (normalized 0-1)
        card_suits = obs['card_suits']
        for suit in card_suits:
            features.append(suit / 3.0)

        return np.array(features, dtype=np.float32)

    def update_target_score(self, target_score: int):
        """Update target score for curriculum learning."""
        self.target_score = target_score


# =============================================================================
# DUELING DQN NETWORK
# =============================================================================

if TORCH_AVAILABLE:
    class DuelingDQN(nn.Module):
        """
        Dueling DQN architecture that separates state value V(s) from
        action advantage A(s,a).

        Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        """

        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
            super(DuelingDQN, self).__init__()

            # Shared feature layers
            self.feature_layer = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )

            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim)
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            """
            Forward pass returning Q-values for all actions.

            Args:
                state: Batch of state tensors (batch_size, state_dim)

            Returns:
                Q-values tensor (batch_size, action_dim)
            """
            features = self.feature_layer(state)

            value = self.value_stream(features)
            advantage = self.advantage_stream(features)

            # Combine: Q = V + (A - mean(A))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

            return q_values


# =============================================================================
# PRIORITIZED EXPERIENCE REPLAY BUFFER
# =============================================================================

class SumTree:
    """
    Binary sum tree for efficient priority-based sampling.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample index given a value s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Return total priority sum."""
        return self.tree[0]

    def add(self, priority: float, data):
        """Add new experience with given priority."""
        idx = self.write_idx + self.capacity - 1

        self.data[self.write_idx] = data
        self.update(idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        """Update priority at given tree index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """Get sample by priority value s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer using sum tree.

    Samples transitions proportional to TD-error priority.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-6):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            epsilon: Small constant to ensure non-zero priorities
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        """Add experience with max priority (will be updated after first learning)."""
        experience = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List, np.ndarray, List[int]]:
        """
        Sample batch with importance sampling weights.

        Args:
            batch_size: Number of samples
            beta: Importance sampling exponent (0 = no correction, 1 = full)

        Returns:
            (experiences, weights, tree_indices)
        """
        batch = []
        indices = []
        priorities = []

        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            idx, priority, data = self.tree.get(s)

            if data is not None and not isinstance(data, (int, float)):
                batch.append(data)
                indices.append(idx)
                priorities.append(priority)

        # If we didn't get enough valid samples, pad with random valid ones
        while len(batch) < batch_size and self.tree.n_entries > 0:
            s = random.uniform(0, self.tree.total())
            idx, priority, data = self.tree.get(s)
            if data is not None and not isinstance(data, (int, float)):
                batch.append(data)
                indices.append(idx)
                priorities.append(priority)

        if not batch:
            return [], np.array([]), []

        # Compute importance sampling weights
        priorities = np.array(priorities)
        probs = priorities / self.tree.total()
        weights = (self.tree.n_entries * probs) ** (-beta)
        weights = weights / weights.max()  # Normalize

        return batch, weights.astype(np.float32), indices

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.tree.n_entries


# =============================================================================
# DUELING DQN AGENT
# =============================================================================

class DuelingDQNAgent:
    """
    Dueling DQN Agent with Prioritized Experience Replay.

    Uses the 5-strategy action space from StrategyActionEncoder.
    """

    NUM_ACTIONS = 5

    def __init__(
        self,
        env: BalatroBatchedSimEnv,
        config: Dict[str, Any],
        device: str = None
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DuelingDQNAgent")

        self.env = env
        self.config = config
        self.action_encoder = StrategyActionEncoder(env)

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # State preprocessor
        self.state_preprocessor = StatePreprocessor(env.target_score)
        self.state_dim = StatePreprocessor.STATE_DIM

        # Network parameters
        hidden_dim = config['network']['hidden_dim']

        # Policy and target networks
        self.policy_net = DuelingDQN(self.state_dim, self.NUM_ACTIONS, hidden_dim).to(self.device)
        self.target_net = DuelingDQN(self.state_dim, self.NUM_ACTIONS, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning']['lr'])

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config['replay']['capacity'],
            alpha=config['replay']['alpha']
        )

        # Learning parameters
        self.gamma = config['learning']['gamma']
        self.tau = config['learning']['tau']
        self.batch_size = config['learning']['batch_size']

        # Exploration parameters
        self.epsilon = config['learning']['epsilon_start']
        self.epsilon_end = config['learning']['epsilon_end']
        self.epsilon_decay = config['learning']['epsilon_decay']

        # Beta annealing for importance sampling
        self.beta = config['replay']['beta_start']
        self.beta_end = config['replay']['beta_end']

        # Tracking
        self._last_action_idx = None
        self.total_steps = 0
        self.total_updates = 0

    def choose_action(self, obs: Dict, explore: bool = True) -> Dict:
        """
        Choose action using epsilon-greedy policy.

        Args:
            obs: Observation dictionary
            explore: Whether to use epsilon-greedy exploration

        Returns:
            Action dictionary with 'type' and 'card_mask'
        """
        valid_actions = self.action_encoder.get_valid_actions(obs)

        if explore and random.random() < self.epsilon:
            # Random action from valid actions
            action_idx = random.choice(valid_actions)
        else:
            # Greedy action
            state = self.state_preprocessor.process(obs)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]

            # Select best valid action
            valid_q_values = {a: q_values[a] for a in valid_actions}
            action_idx = max(valid_q_values, key=valid_q_values.get)

        self._last_action_idx = action_idx
        return self.action_encoder.decode_to_env_action(action_idx, obs)

    def store_transition(self, obs, action_idx, reward, next_obs, done):
        """Store transition in replay buffer with reward normalization."""
        state = self.state_preprocessor.process(obs)
        next_state = self.state_preprocessor.process(next_obs)
        # Normalize rewards to prevent exploding Q-values
        # Typical rewards range from -500 (loss) to +1000 (win)
        # Scale to roughly [-1, 1] range
        normalized_reward = reward / 100.0
        self.replay_buffer.push(state, action_idx, normalized_reward, next_state, done)
        self.total_steps += 1

    def update(self) -> Optional[float]:
        """
        Perform one gradient update step.

        Returns:
            Loss value or None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        batch, weights, indices = self.replay_buffer.sample(self.batch_size, self.beta)

        if not batch:
            return None

        # Unpack batch
        states = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        actions = torch.LongTensor([t[1] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in batch]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN style)
        with torch.no_grad():
            # Select actions using policy network
            next_actions = self.policy_net(next_states).argmax(1)
            # Evaluate using target network
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # TD errors for priority update
        td_errors = (current_q - target_q).detach().cpu().numpy()

        # Weighted Huber loss (more robust to outliers than MSE)
        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

        # Optimize with tighter gradient clipping to prevent instability
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)

        # Soft update target network
        self._soft_update()

        self.total_updates += 1

        return loss.item()

    def _soft_update(self):
        """Soft update target network parameters."""
        for target_param, policy_param in zip(self.target_net.parameters(),
                                               self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def anneal_beta(self, progress: float):
        """Anneal importance sampling beta based on training progress."""
        self.beta = self.config['replay']['beta_start'] + \
                   progress * (self.beta_end - self.config['replay']['beta_start'])

    def update_target_score(self, target_score: int):
        """Update target score for curriculum learning."""
        self.state_preprocessor.update_target_score(target_score)

    def get_last_action_name(self) -> str:
        """Get name of last chosen action for logging."""
        if self._last_action_idx is not None:
            return self.action_encoder.get_action_name(self._last_action_idx)
        return "UNKNOWN"

    def save_checkpoint(self, filepath: str, metadata: Dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'beta': self.beta,
            'total_steps': self.total_steps,
            'total_updates': self.total_updates,
            'config': self.config,
            'metadata': metadata or {}
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.beta = checkpoint.get('beta', self.config['replay']['beta_start'])
        self.total_steps = checkpoint['total_steps']
        self.total_updates = checkpoint['total_updates']

        return checkpoint.get('metadata', {})


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train(
    agent: DuelingDQNAgent,
    env: BalatroBatchedSimEnv,
    config: Dict,
    start_episode: int = 0
) -> Dict[str, List]:
    """
    Train Dueling DQN agent with curriculum learning.
    """
    curriculum_enabled = config['curriculum']['enabled']
    phases = config['curriculum']['phases'] if curriculum_enabled else [
        {'target': env.target_score, 'episodes': config['training']['num_episodes']}
    ]

    eval_freq = config['training']['eval_frequency']
    eval_episodes = config['training']['eval_episodes']
    checkpoint_freq = config['training']['checkpoint_frequency']
    max_steps = config['training']['max_steps_per_episode']
    verbose = config['logging']['verbose']
    log_freq = config['logging']['log_frequency']

    train_stats = {
        'episodes': [],
        'win_rates': [],
        'avg_rewards': [],
        'epsilons': [],
        'losses': []
    }

    # Evaluate random baseline
    if config['baseline']['compare_to_random']:
        print("\nEvaluating random strategy baseline...")
        baseline_win_rate, baseline_avg_reward = evaluate_random_strategy_baseline(
            env, config['baseline']['random_episodes'], seed=42
        )
        print(f"Random Baseline: Win Rate = {baseline_win_rate:.1%}, "
              f"Avg Reward = {baseline_avg_reward:.1f}")

    global_episode = start_episode
    total_episodes = sum(phase['episodes'] for phase in phases)

    print(f"\nStarting Dueling DQN training...")
    print(f"Total episodes across curriculum: {total_episodes}")
    print(f"Device: {agent.device}")
    print(f"Initial epsilon: {agent.epsilon:.3f}\n")

    for phase_idx, phase in enumerate(phases):
        target_score = phase['target']
        phase_episodes = phase['episodes']

        # Update environment and agent for new target
        env.target_score = target_score
        agent.update_target_score(target_score)

        print(f"\n{'='*60}")
        print(f"CURRICULUM PHASE {phase_idx + 1}/{len(phases)}")
        print(f"Target Score: {target_score}")
        print(f"Episodes: {phase_episodes}")
        print(f"{'='*60}\n")

        for episode in range(phase_episodes):
            obs, _ = env.reset(seed=global_episode)
            episode_reward = 0.0
            episode_loss = 0.0
            loss_count = 0
            steps = 0
            done = False

            while not done and steps < max_steps:
                # Choose action
                action = agent.choose_action(obs, explore=True)
                action_idx = agent._last_action_idx

                # Execute action
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store transition
                agent.store_transition(obs, action_idx, reward, next_obs, done)

                # Learn
                loss = agent.update()
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1

                episode_reward += reward
                obs = next_obs
                steps += 1

            # Decay epsilon
            agent.decay_epsilon()

            # Anneal beta
            progress = global_episode / total_episodes
            agent.anneal_beta(progress)

            # Logging
            if verbose and (global_episode + 1) % log_freq == 0:
                avg_loss = episode_loss / max(loss_count, 1)
                win_status = "WIN" if info.get('win', False) else "LOSS"
                print(f"Episode {global_episode + 1:5d} (T={target_score}): "
                      f"Reward = {episode_reward:7.1f}, Steps = {steps:2d}, {win_status}, "
                      f"Loss = {avg_loss:.4f}, Epsilon = {agent.epsilon:.3f}")

            # Evaluation
            if (global_episode + 1) % eval_freq == 0:
                print(f"\nEvaluating agent at target={target_score}...")
                win_rate, avg_reward = evaluate(agent, env, eval_episodes,
                                                seed=global_episode + 10000)

                train_stats['episodes'].append(global_episode + 1)
                train_stats['win_rates'].append(win_rate)
                train_stats['avg_rewards'].append(avg_reward)
                train_stats['epsilons'].append(agent.epsilon)
                train_stats['losses'].append(episode_loss / max(loss_count, 1))

                print(f"\n{'='*60}")
                print(f"Evaluation at Episode {global_episode + 1} (Target: {target_score})")
                print(f"{'='*60}")
                print(f"  Win Rate:     {win_rate:5.1%}")
                print(f"  Avg Reward:   {avg_reward:7.1f}")
                print(f"  Epsilon:      {agent.epsilon:.3f}")
                print(f"  Beta:         {agent.beta:.3f}")
                print(f"  Buffer Size:  {len(agent.replay_buffer)}")
                print(f"{'='*60}\n")

            # Checkpointing
            if (global_episode + 1) % checkpoint_freq == 0:
                script_dir = Path(__file__).parent
                model_dir = script_dir / config['model']['save_dir']
                filename = create_model_filename("dueling_dqn", global_episode + 1)
                filepath = model_dir / filename

                metadata = {
                    'episode': global_episode + 1,
                    'phase': phase_idx,
                    'target_score': target_score,
                    'train_stats': train_stats
                }

                agent.save_checkpoint(str(filepath), metadata)
                print(f"Checkpoint saved to {filepath}")

            global_episode += 1

    print("\nTraining complete!")
    print(f"Total episodes: {global_episode}")
    print(f"Total steps: {agent.total_steps}")
    print(f"Total updates: {agent.total_updates}")

    return train_stats


def evaluate(
    agent: DuelingDQNAgent,
    env: BalatroBatchedSimEnv,
    num_episodes: int,
    seed: int = 42,
    visualize: bool = False,
    viz_mode: str = 'full'
) -> Tuple[float, float]:
    """Evaluate agent performance without exploration."""
    wins = 0
    total_reward = 0.0

    visualizer = None
    if visualize:
        visualizer = AgentVisualizer()

    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed + i)
        episode_reward = 0.0
        done = False
        steps = 0
        max_steps = 100

        if visualize:
            visualizer.reset_episode()
            visualizer.print_episode_header(i + 1, env.target_score, seed=seed + i)

        while not done and steps < max_steps:
            action = agent.choose_action(obs, explore=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if visualize:
                action_name = agent.get_last_action_name()
                if viz_mode == 'compact':
                    visualizer.visualize_step_compact(obs, action, next_obs, reward, info,
                                                      decision=action_name)
                else:
                    visualizer.visualize_step(obs, action, next_obs, reward, info,
                                             decision=action_name)

            obs = next_obs
            steps += 1

            if done and info.get('win', False):
                wins += 1

        total_reward += episode_reward

        if visualize:
            result = {
                'win': info.get('win', False),
                'steps': steps,
                'total_reward': episode_reward,
                'final_chips': info.get('chips', 0)
            }
            visualizer.print_episode_summary(result)

        if not visualize and (i + 1) % 5 == 0:
            print(".", end="", flush=True)

    if not visualize:
        print()

    return wins / num_episodes, total_reward / num_episodes


def evaluate_dueling_dqn_baseline(env, agent, num_episodes: int, seed: int = 42) -> Tuple[float, float]:
    """
    Evaluate Dueling DQN agent for comparison with other agents.

    Returns:
        Tuple of (win_rate, avg_reward)
    """
    return evaluate(agent, env, num_episodes, seed=seed, visualize=False)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dueling DQN with Prioritized Replay and Curriculum Learning for Balatro Poker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--config", type=str, default="dueling_dqn_config.yaml")
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-score", type=int, default=300)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--viz-mode", choices=["full", "compact"], default="full")

    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required. Install with: pip install torch")
        return 1

    # Load config
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with CLI args
    if args.episodes is not None:
        config['training']['num_episodes'] = args.episodes
    if args.lr is not None:
        config['learning']['lr'] = args.lr

    target_score = args.target_score
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create environment
    env = BalatroBatchedSimEnv(target_score=target_score)

    # Create agent
    agent = DuelingDQNAgent(env, config)

    # Load checkpoint if specified
    start_episode = 0
    if args.load_model:
        script_dir = Path(__file__).parent
        model_path = Path(args.load_model)
        if not model_path.is_absolute():
            model_path = script_dir / model_path

        print(f"Loading model from {model_path}...")
        metadata = agent.load_checkpoint(str(model_path))
        start_episode = metadata.get('episode', 0)
        print(f"  Resumed from episode {start_episode}")
        print(f"  Epsilon: {agent.epsilon:.3f}")

    print("=" * 60)
    print("DUELING DQN AGENT FOR BALATRO POKER")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Target Score: {target_score}")
    print(f"Device: {agent.device}")
    print()
    print("Architecture:")
    print("  - Dueling DQN (separate V and A streams)")
    print("  - Prioritized Experience Replay")
    print("  - Curriculum Learning" if config['curriculum']['enabled'] else "  - Single target training")
    print()
    print(f"Network: {StatePreprocessor.STATE_DIM} -> 128 -> 128 -> (V + A)")
    print(f"Actions: 5 strategies")
    print(f"Replay Buffer: {config['replay']['capacity']}")
    print("=" * 60)

    if args.mode == "train":
        train_stats = train(agent, env, config, start_episode)

        # Final comparison
        if config['baseline']['compare_to_random']:
            print("\nFinal Evaluation...")
            final_win_rate, final_avg_reward = evaluate(
                agent, env, config['baseline']['random_episodes'], seed=99999
            )
            baseline_win_rate, baseline_avg_reward = evaluate_random_strategy_baseline(
                env, config['baseline']['random_episodes'], seed=42
            )
            compare_to_baseline(
                final_win_rate, final_avg_reward,
                baseline_win_rate, baseline_avg_reward
            )

        # Plot training curves
        if args.plot and train_stats['episodes']:
            script_dir = Path(__file__).parent
            plot_path = script_dir / config['model']['save_dir'] / "dueling_dqn_training_curves.png"
            plot_training_curves(
                train_stats['episodes'],
                train_stats['win_rates'],
                train_stats['avg_rewards'],
                train_stats['epsilons'],
                save_path=plot_path
            )

    elif args.mode == "eval":
        num_eval_episodes = config['training']['eval_episodes']
        visualize = args.visualize
        viz_mode = args.viz_mode

        if visualize:
            print(f"\nEvaluating with visualization ({viz_mode} mode)...")
        else:
            print(f"\nEvaluating over {num_eval_episodes} episodes...")

        win_rate, avg_reward = evaluate(
            agent, env, num_eval_episodes,
            seed=args.seed, visualize=visualize, viz_mode=viz_mode
        )

        if not visualize:
            print(f"\nEvaluation Results:")
            print(f"  Win Rate:   {win_rate:5.1%}")
            print(f"  Avg Reward: {avg_reward:7.1f}")

        if config['baseline']['compare_to_random'] and not visualize:
            baseline_win_rate, baseline_avg_reward = evaluate_random_strategy_baseline(
                env, num_eval_episodes, seed=42
            )
            compare_to_baseline(
                win_rate, avg_reward,
                baseline_win_rate, baseline_avg_reward
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
