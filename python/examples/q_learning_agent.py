#!/usr/bin/env python3
"""
Tabular Q-Learning Agent for Balatro Poker Environment.

Implements classic Q-learning with strategy-based action space.
Uses 5 high-level strategies instead of 512 card mask combinations.
"""

import sys
import os
import argparse
import yaml
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List, Any

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from balatro_env import BalatroBatchedSimEnv
from q_learning_utils import (
    save_checkpoint, load_checkpoint, plot_training_curves,
    compare_to_baseline, print_q_table_stats, create_model_filename
)
from random_strategy_agent import evaluate_random_strategy_baseline
from agent_visualizer import AgentVisualizer
from strategy_action_encoder import StrategyActionEncoder


class StateDiscretizer:
    """
    State discretization with draw indicators for hand improvement potential.

    Uses 8 compact features with specific draw indicators to help the model
    learn when discarding is worthwhile.

    Features (8 total):
    1. plays_left (0-4) - resource tracking
    2. discards_left (0-3) - resource tracking
    3. chips_bucket (0-11) - progress tracking
    4. best_hand_type (0-8) - current hand strength
    5. can_win_this_play (0-1) - can we win with current hand?
    6. flush_draw (0-1) - has 4+ cards same suit (not already flush)
    7. straight_draw (0-2) - 0=none, 1=gutshot, 2=open-ended
    8. upgrade_potential (0-2) - room for improvement based on hand type

    State space: 5 × 4 × 12 × 9 × 2 × 2 × 3 × 3 = 38,880 theoretical states
    """

    def __init__(self, config: Dict[str, Any], target_score: int):
        """
        Initialize with finer-grained chip discretization.

        Args:
            config: Discretization configuration (chips_bins used if present)
            target_score: Target score for the environment
        """
        self.target_score = target_score

        # Use finer chips discretization with more granularity near goal
        self.chips_bins = np.array([
            0.0,   # 0%
            0.20,  # 20%
            0.40,  # 40%
            0.60,  # 60%
            0.75,  # 75% - threshold bonus
            0.85,  # 85%
            0.90,  # 90% - threshold bonus
            0.95,  # 95% - very close
            1.00,  # 100% - at goal
            1.10,  # 110% - safely past
            1.25   # 125%+ - way past
        ])

    def _has_flush_draw(self, obs: Dict) -> int:
        """
        Check if we have a flush draw (4+ cards same suit, not already flush).

        Returns:
            0 = No flush draw or already have flush
            1 = Has flush draw (4+ suited cards)
        """
        if obs.get('has_flush', False):
            return 0  # Already have flush, not a "draw"
        return 1 if obs.get('flush_potential', False) else 0

    def _get_straight_draw_type(self, obs: Dict) -> int:
        """
        Determine the type of straight draw.

        Returns:
            0 = No straight draw or already have straight
            1 = Gutshot (4 outs, one card needed inside)
            2 = Open-ended (8 outs, can complete on either end)
        """
        if obs.get('has_straight', False):
            return 0  # Already have straight

        # Get unique sorted ranks
        ranks = sorted(set(obs['card_ranks']))

        if len(ranks) < 4:
            return 0  # Not enough unique ranks

        # Check for open-ended: 4 consecutive ranks
        for i in range(len(ranks) - 3):
            if ranks[i+3] - ranks[i] == 3:  # 4 consecutive
                # Open-ended if not at edges (can complete on both ends)
                # Not open-ended: A-2-3-4 (can only get 5) or J-Q-K-A (can only get 10)
                if ranks[i] > 0 and ranks[i+3] < 12:
                    return 2  # Open-ended straight draw

        # Check for gutshot: 4 cards with one gap
        for i in range(len(ranks) - 3):
            span = ranks[i+3] - ranks[i]
            if span == 4:  # One gap in 4 cards
                # Verify it's actually a gutshot (3 consecutive + 1 gap)
                gaps = [ranks[j+1] - ranks[j] for j in range(i, i+3)]
                if gaps.count(2) == 1 and gaps.count(1) == 2:
                    return 1  # Gutshot straight draw

        return 0  # No straight draw

    def _get_upgrade_potential(self, obs: Dict) -> int:
        """
        Determine upgrade potential based on current hand type.

        Returns:
            0 = Low potential (already strong: FLUSH, FULL_HOUSE, FOUR_OF_A_KIND, STRAIGHT_FLUSH)
            1 = Medium potential (decent hand: PAIR, TWO_PAIR, STRAIGHT)
            2 = High potential (weak/upgradeable: HIGH_CARD, THREE_OF_A_KIND)

        Note: THREE_OF_A_KIND is marked high because trips can become:
        - Full house (drawing to any pair)
        - Four of a kind (drawing to the remaining card)
        """
        hand_type = obs['best_hand_type']

        # Hand types: 0=HIGH_CARD, 1=PAIR, 2=TWO_PAIR, 3=THREE_OF_A_KIND,
        # 4=STRAIGHT, 5=FLUSH, 6=FULL_HOUSE, 7=FOUR_OF_A_KIND, 8=STRAIGHT_FLUSH
        if hand_type >= 5:  # FLUSH or better
            return 0  # Low - already strong
        elif hand_type in [1, 2, 4]:  # PAIR, TWO_PAIR, STRAIGHT
            return 1  # Medium - decent but could improve
        else:  # HIGH_CARD (0) or THREE_OF_A_KIND (3)
            return 2  # High - significant improvement possible

    def discretize(self, obs: Dict) -> Tuple:
        """
        Convert observation to discrete state with draw indicators.

        Returns 8-feature state:
        1. plays_left (0-4) - resource tracking
        2. discards_left (0-3) - resource tracking
        3. chips_bucket (0-11) - progress tracking
        4. best_hand_type (0-8) - C++ evaluated hand strength
        5. can_win_this_play (0-1) - can we win now?
        6. flush_draw (0-1) - has flush draw?
        7. straight_draw (0-2) - type of straight draw
        8. upgrade_potential (0-2) - room for improvement

        Args:
            obs: Observation dictionary with C++ enhanced fields

        Returns:
            Hashable 8-element tuple representing discrete state
        """
        plays_left = obs['plays_left']
        discards_left = obs['discards_left']

        # Chips discretization
        chips = obs['chips'][0]
        chips_progress = chips / self.target_score
        chips_bucket = np.digitize(chips_progress, self.chips_bins)

        # Use C++ best hand type directly
        best_hand_type = obs['best_hand_type']  # 0-8 from C++

        # Can win estimate using C++ best_hand_score
        best_hand_score = obs['best_hand_score'][0]
        chips_needed = self.target_score - chips
        can_win_this_play = int(best_hand_score >= chips_needed)

        # NEW: Draw indicators
        flush_draw = self._has_flush_draw(obs)
        straight_draw = self._get_straight_draw_type(obs)
        upgrade_potential = self._get_upgrade_potential(obs)

        state = (
            plays_left,
            discards_left,
            chips_bucket,
            best_hand_type,
            can_win_this_play,
            flush_draw,
            straight_draw,
            upgrade_potential
        )

        return state


# ActionEncoder class replaced by StrategyActionEncoder (imported above)
# Old action space: 512 card mask combinations
# New action space: 5 strategic actions (PLAY_BEST_HAND, DISCARD_UPGRADE, etc.)


class QLearningAgent:
    """
    Tabular Q-Learning agent with strategy-based action space.

    Uses 5 high-level strategies instead of 512 card mask combinations:
    - 0: PLAY_BEST_HAND
    - 1: DISCARD_UPGRADE
    - 2: DISCARD_FLUSH_CHASE
    - 3: DISCARD_STRAIGHT_CHASE
    - 4: DISCARD_AGGRESSIVE
    """

    def __init__(
        self,
        env: BalatroBatchedSimEnv,
        state_discretizer: StateDiscretizer,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        default_q_value: float = 0.0
    ):
        """
        Initialize Q-learning agent with strategy-based actions.

        Args:
            env: Environment instance (needed for strategy encoder)
            state_discretizer: State discretization object
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay per episode
            default_q_value: Default Q-value for unseen state-action pairs
        """
        self.env = env
        self.state_discretizer = state_discretizer
        self.action_encoder = StrategyActionEncoder(env)

        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.default_q = default_q_value

        # Q-table: Q[state][strategy_idx] = q_value
        # Now only 5 actions per state instead of 512!
        self.q_table = defaultdict(lambda: defaultdict(lambda: self.default_q))

        # Track last chosen strategy for update
        self._last_strategy_idx = None

        # Statistics
        self.total_updates = 0

    def get_q_value(self, state: Tuple, action_idx: int) -> float:
        """Get Q-value for state-action pair."""
        return self.q_table[state][action_idx]

    def set_q_value(self, state: Tuple, action_idx: int, value: float):
        """Set Q-value for state-action pair."""
        self.q_table[state][action_idx] = value

    def get_max_q_value(self, state: Tuple, obs: Dict) -> float:
        """Get maximum Q-value for state across valid actions."""
        valid_actions = self.get_valid_action_indices(obs)

        if state not in self.q_table or not self.q_table[state]:
            return self.default_q

        # Only consider valid actions
        valid_q_values = [self.q_table[state].get(a, self.default_q) for a in valid_actions]
        return max(valid_q_values) if valid_q_values else self.default_q

    def get_valid_action_indices(self, obs: Dict) -> List[int]:
        """
        Get list of valid strategy action indices for current observation.

        Returns:
            List of valid action indices (0-4)
        """
        return self.action_encoder.get_valid_actions(obs)

    def get_best_action(self, state: Tuple, obs: Dict) -> int:
        """
        Get strategy action with highest Q-value for state (exploitation).

        Args:
            state: Discretized state tuple
            obs: Raw observation (for action filtering)

        Returns:
            Best valid strategy index (0-4)
        """
        valid_actions = self.get_valid_action_indices(obs)

        if state not in self.q_table or not self.q_table[state]:
            # No Q-values yet, return random valid action
            return np.random.choice(valid_actions)

        # Filter Q-values to only include valid actions
        valid_q_values = {a: self.q_table[state].get(a, self.default_q)
                         for a in valid_actions}

        if not valid_q_values:
            return np.random.choice(valid_actions)

        # Return valid action with max Q-value
        return max(valid_q_values.items(), key=lambda x: x[1])[0]

    def choose_action(self, obs: Dict, explore: bool = True) -> Dict:
        """
        Choose action using epsilon-greedy policy over strategies.

        Args:
            obs: Observation from environment
            explore: Whether to use epsilon-greedy (False for evaluation)

        Returns:
            Action dict with 'type' and 'card_mask'
        """
        state = self.state_discretizer.discretize(obs)
        valid_actions = self.get_valid_action_indices(obs)

        # Epsilon-greedy over strategy indices
        if explore and np.random.random() < self.epsilon:
            strategy_idx = np.random.choice(valid_actions)
        else:
            strategy_idx = self.get_best_action(state, obs)

        # Store for update
        self._last_strategy_idx = strategy_idx

        # Convert strategy to environment action
        action = self.action_encoder.decode_to_env_action(strategy_idx, obs)

        return action

    def get_last_strategy_name(self) -> str:
        """Get name of last chosen strategy for logging."""
        if self._last_strategy_idx is not None:
            return self.action_encoder.get_action_name(self._last_strategy_idx)
        return "UNKNOWN"

    def update(self, obs: Dict, action: Dict, reward: float, next_obs: Dict, done: bool):
        """
        Update Q-value using Q-learning update rule.

        Q(s,a) ← Q(s,a) + α[r + γ·max_a'(Q(s',a')) - Q(s,a)]

        Args:
            obs: Current observation
            action: Action taken (not used - we use stored strategy_idx)
            reward: Reward received
            next_obs: Next observation
            done: Whether episode ended
        """
        state = self.state_discretizer.discretize(obs)
        strategy_idx = self._last_strategy_idx
        next_state = self.state_discretizer.discretize(next_obs)

        if strategy_idx is None:
            return  # No action was taken

        # Current Q-value
        current_q = self.get_q_value(state, strategy_idx)

        # Target Q-value
        if done:
            target_q = reward  # No future rewards
        else:
            max_next_q = self.get_max_q_value(next_state, next_obs)
            target_q = reward + self.gamma * max_next_q

        # Q-learning update
        new_q = current_q + self.alpha * (target_q - current_q)
        self.set_q_value(state, strategy_idx, new_q)

        self.total_updates += 1

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train(
    agent: QLearningAgent,
    env: BalatroBatchedSimEnv,
    config: Dict,
    start_episode: int = 0
) -> Dict[str, List]:
    """
    Train Q-learning agent.

    Args:
        agent: Q-learning agent
        env: Balatro environment
        config: Training configuration
        start_episode: Starting episode number (for resuming)

    Returns:
        Dictionary with training statistics
    """
    num_episodes = config['training']['num_episodes']
    eval_freq = config['training']['eval_frequency']
    eval_episodes = config['training']['eval_episodes']
    checkpoint_freq = config['training']['checkpoint_frequency']
    verbose = config['logging']['verbose']
    log_freq = config['logging']['log_frequency']

    # Statistics
    train_stats = {
        'episodes': [],
        'win_rates': [],
        'avg_rewards': [],
        'epsilons': []
    }

    # Evaluate random strategy baseline once (same action space for fair comparison)
    if config['baseline']['compare_to_random'] and start_episode == 0:
        print("\nEvaluating random strategy baseline (same action space)...")
        baseline_episodes = config['baseline']['random_episodes']
        baseline_win_rate, baseline_avg_reward = evaluate_random_strategy_baseline(
            env, baseline_episodes, seed=42
        )
        print(f"Random Strategy Baseline: Win Rate = {baseline_win_rate:.1%}, "
              f"Avg Reward = {baseline_avg_reward:.1f}")

    print(f"\nStarting training from episode {start_episode}...")
    print(f"Target episodes: {num_episodes}")
    print(f"Initial epsilon: {agent.epsilon:.3f}\n")

    for episode in range(start_episode, num_episodes):
        obs, _ = env.reset(seed=episode)
        episode_reward = 0.0
        steps = 0
        done = False
        max_steps = 100  # Safety limit to prevent infinite loops

        while not done and steps < max_steps:
            # Choose and execute action
            action = agent.choose_action(obs, explore=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update Q-values
            agent.update(obs, action, reward, next_obs, done)

            episode_reward += reward
            obs = next_obs
            steps += 1

        # Debug: Warn if we hit max steps
        if steps >= max_steps and not done:
            print(f"\n[WARNING] Episode {episode+1} hit max steps limit ({max_steps})")
            print(f"  Agent may be choosing invalid actions repeatedly")

        # Decay epsilon
        agent.decay_epsilon()

        # Logging
        if verbose and (episode + 1) % log_freq == 0:
            win_status = "WIN" if info.get('win', False) else "LOSS"
            print(f"Episode {episode + 1:5d}: Reward = {episode_reward:7.1f}, "
                  f"Steps = {steps:2d}, {win_status}, Epsilon = {agent.epsilon:.3f}")

        # Evaluation
        if (episode + 1) % eval_freq == 0:
            print(f"\nEvaluating agent (running {eval_episodes} episodes without exploration)...")
            win_rate, avg_reward = evaluate(agent, env, eval_episodes, seed=episode + 10000)

            train_stats['episodes'].append(episode + 1)
            train_stats['win_rates'].append(win_rate)
            train_stats['avg_rewards'].append(avg_reward)
            train_stats['epsilons'].append(agent.epsilon)

            print(f"\n{'=' * 60}")
            print(f"Evaluation at Episode {episode + 1}")
            print(f"{'=' * 60}")
            print(f"  Win Rate:     {win_rate:5.1%}")
            print(f"  Avg Reward:   {avg_reward:7.1f}")
            print(f"  Q-Table Size: {len(agent.q_table)} states")
            print(f"  Epsilon:      {agent.epsilon:.3f}")
            print(f"{'=' * 60}\n")

        # Checkpointing
        if (episode + 1) % checkpoint_freq == 0:
            model_dir = config['model']['save_dir']
            model_prefix = config['model']['model_prefix']
            filename = create_model_filename(model_prefix, episode + 1)
            filepath = os.path.join(model_dir, filename)

            metadata = {
                'episode': episode + 1,
                'epsilon': agent.epsilon,
                'total_updates': agent.total_updates,
                'config': config
            }

            # Convert nested defaultdict to regular dict for pickling
            q_table_dict = {state: dict(actions) for state, actions in agent.q_table.items()}
            save_checkpoint(q_table_dict, metadata, filepath)

    print("\nTraining complete!")
    print_q_table_stats(dict(agent.q_table))

    return train_stats


def evaluate(
    agent: QLearningAgent,
    env: BalatroBatchedSimEnv,
    num_episodes: int,
    seed: int = 42,
    visualize: bool = False,
    viz_mode: str = 'full'
) -> Tuple[float, float]:
    """
    Evaluate agent performance without exploration.

    Args:
        agent: Q-learning agent
        env: Balatro environment
        num_episodes: Number of evaluation episodes
        seed: Random seed
        visualize: If True, show agent decisions with card details
        viz_mode: 'full' for detailed view, 'compact' for one-line summaries

    Returns:
        (win_rate, avg_reward) tuple
    """
    wins = 0
    total_reward = 0.0

    # Create visualizer if requested
    visualizer = None
    if visualize:
        visualizer = AgentVisualizer()

    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed + i)
        episode_reward = 0.0
        done = False
        steps = 0
        max_steps = 100  # Safety limit to prevent infinite loops

        # Print episode header if visualizing
        if visualize:
            visualizer.reset_episode()
            visualizer.print_episode_header(i + 1, env.target_score, seed=seed + i)

        while not done and steps < max_steps:
            # Choose action without exploration
            action = agent.choose_action(obs, explore=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Visualize decision
            if visualize:
                if viz_mode == 'compact':
                    visualizer.visualize_step_compact(obs, action, next_obs, reward, info)
                else:  # full mode
                    visualizer.visualize_step(obs, action, next_obs, reward, info)

            obs = next_obs
            steps += 1

            if done and info.get('win', False):
                wins += 1

        total_reward += episode_reward

        # Print episode summary if visualizing
        if visualize:
            result = {
                'win': info.get('win', False),
                'steps': steps,
                'total_reward': episode_reward,
                'final_chips': info.get('chips', 0)
            }
            visualizer.print_episode_summary(result)

        # Debug: Warn if we hit max steps
        if steps >= max_steps and not done:
            print(f"\n[WARNING] Eval episode {i+1} hit max steps limit ({max_steps})")
            print(f"  This suggests the agent is stuck in an invalid action loop")

        # Progress indicator (print dot every 5 episodes) - skip if visualizing
        if not visualize and (i + 1) % 5 == 0:
            print(".", end="", flush=True)

    if not visualize:
        print()  # New line after progress dots

    win_rate = wins / num_episodes
    avg_reward = total_reward / num_episodes

    return win_rate, avg_reward


def main():
    parser = argparse.ArgumentParser(
        description="Tabular Q-Learning for Balatro Poker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="Mode: train a new model or evaluate existing model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="q_learning_config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to checkpoint file to load (for resuming or evaluation)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of episodes from config"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config"
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=None,
        help="Override initial epsilon from config"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--target-score",
        type=int,
        default=300,
        help="Target score for environment"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot training curves after training"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show agent decisions with card details (for eval mode)"
    )
    parser.add_argument(
        "--viz-mode",
        choices=["full", "compact"],
        default="full",
        help="Visualization mode: 'full' for detailed view, 'compact' for one-line summaries"
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command-line arguments
    if args.episodes is not None:
        config['training']['num_episodes'] = args.episodes
    if args.lr is not None:
        config['learning']['alpha'] = args.lr
    if args.epsilon_start is not None:
        config['learning']['epsilon_start'] = args.epsilon_start

    target_score = args.target_score
    np.random.seed(args.seed)

    # Create environment
    reward_config = config['environment'].get('reward_config')
    env = BalatroBatchedSimEnv(
        target_score=target_score,
        reward_config_path=reward_config
    )

    # Create state discretizer (action encoder is created inside agent)
    state_discretizer = StateDiscretizer(
        config['state_discretization'],
        target_score
    )

    # Create or load agent
    if args.load_model:
        print(f"Loading model from {args.load_model}...")
        q_table_dict, metadata = load_checkpoint(args.load_model)

        # Convert dict back to defaultdict
        q_table = defaultdict(lambda: defaultdict(lambda: config['q_table']['default_value']))
        for state, actions in q_table_dict.items():
            q_table[state] = defaultdict(lambda: config['q_table']['default_value'], actions)

        agent = QLearningAgent(
            env,
            state_discretizer,
            learning_rate=config['learning']['alpha'],
            discount_factor=config['learning']['gamma'],
            epsilon_start=metadata.get('epsilon', config['learning']['epsilon_start']),
            epsilon_end=config['learning']['epsilon_end'],
            epsilon_decay=config['learning']['epsilon_decay'],
            default_q_value=config['q_table']['default_value']
        )
        agent.q_table = q_table
        agent.total_updates = metadata.get('total_updates', 0)

        start_episode = metadata.get('episode', 0)
    else:
        agent = QLearningAgent(
            env,
            state_discretizer,
            learning_rate=config['learning']['alpha'],
            discount_factor=config['learning']['gamma'],
            epsilon_start=config['learning']['epsilon_start'],
            epsilon_end=config['learning']['epsilon_end'],
            epsilon_decay=config['learning']['epsilon_decay'],
            default_q_value=config['q_table']['default_value']
        )
        start_episode = 0

    print("=" * 60)
    print("Q-LEARNING AGENT FOR BALATRO POKER")
    print("(Strategy-Based Action Space)")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Target Score: {target_score}")
    print(f"Action Space: {agent.action_encoder.NUM_ACTIONS} strategies")
    print(f"  - PLAY_BEST_HAND, DISCARD_UPGRADE, DISCARD_FLUSH_CHASE,")
    print(f"    DISCARD_STRAIGHT_CHASE, DISCARD_AGGRESSIVE")
    print(f"Learning Rate: {agent.alpha}")
    print(f"Discount Factor: {agent.gamma}")
    print(f"Epsilon: {agent.epsilon:.3f}")
    print(f"Q-Table Size: {len(agent.q_table)} states")
    print("=" * 60)

    if args.mode == "train":
        # Train agent
        train_stats = train(agent, env, config, start_episode)

        # Compare to strategy baseline (fair comparison with same action space)
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
            plot_path = os.path.join(config['model']['save_dir'], "training_curves.png")
            plot_training_curves(
                train_stats['episodes'],
                train_stats['win_rates'],
                train_stats['avg_rewards'],
                train_stats['epsilons'],
                save_path=plot_path
            )

    elif args.mode == "eval":
        # Evaluate agent
        num_eval_episodes = config['training']['eval_episodes']
        visualize = args.visualize
        viz_mode = args.viz_mode

        if visualize:
            print(f"\nEvaluating agent over {num_eval_episodes} episodes with visualization ({viz_mode} mode)...")
        else:
            print(f"\nEvaluating agent over {num_eval_episodes} episodes...")

        win_rate, avg_reward = evaluate(
            agent, env, num_eval_episodes,
            seed=args.seed,
            visualize=visualize,
            viz_mode=viz_mode
        )

        if not visualize:  # Summary already printed in visualize mode
            print(f"\nEvaluation Results:")
            print(f"  Win Rate:   {win_rate:5.1%}")
            print(f"  Avg Reward: {avg_reward:7.1f}")

        # Compare to strategy baseline (fair comparison with same action space)
        if config['baseline']['compare_to_random'] and not visualize:
            baseline_win_rate, baseline_avg_reward = evaluate_random_strategy_baseline(
                env, num_eval_episodes, seed=42
            )
            compare_to_baseline(
                win_rate, avg_reward,
                baseline_win_rate, baseline_avg_reward
            )


if __name__ == "__main__":
    main()
