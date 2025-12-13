#!/usr/bin/env python3
"""
Hierarchical Q-Learning Agent for Balatro Poker Environment.

Uses two Q-tables for hierarchical decision making:
    1. Q_policy: Learns when to PLAY vs DISCARD (2 actions)
    2. Q_strategy: Learns which discard strategy to use (4 actions)

Architecture:
    Q_policy(state) -> PLAY or DISCARD?
         |                    |
         v                    v
    PLAY_BEST_HAND      Q_strategy(state) -> Which discard?
                             |
                             v
                   [UPGRADE, FLUSH_CHASE,
                    STRAIGHT_CHASE, AGGRESSIVE]

State Features (8 dimensions, ~39K total states):
    1. plays_left (0-4): Remaining plays
    2. discards_left (0-3): Remaining discards
    3. chips_bucket (0-10): Progress toward target score
    4. best_hand_type (0-8): Current best hand (from C++)
    5. can_win_this_play (0-1): Whether best hand score >= remaining chips
    6. flush_draw (0-2): Flush draw status (0=none, 1=draw, 2=have)
    7. straight_draw (0-2): Straight draw type (0=none, 1=gutshot, 2=open)
    8. upgrade_potential (0-2): How much hand could improve

Hyperparameters (from q_learning_config.yaml):
    - alpha: 0.1 (learning rate)
    - gamma: 0.95 (discount factor)
    - epsilon_start: 1.0 -> epsilon_min: 0.05 (exploration)
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


# =============================================================================
# HIERARCHICAL ACTION CONSTANTS
# =============================================================================

# Policy-level actions (Q_policy)
POLICY_PLAY = 0
POLICY_DISCARD = 1
POLICY_ACTIONS = [POLICY_PLAY, POLICY_DISCARD]
POLICY_NAMES = ["PLAY", "DISCARD"]

# Strategy-level actions (Q_strategy) - only used when DISCARD is chosen
STRATEGY_UPGRADE = 0      # Maps to StrategyActionEncoder action 1
STRATEGY_FLUSH = 1        # Maps to StrategyActionEncoder action 2
STRATEGY_STRAIGHT = 2     # Maps to StrategyActionEncoder action 3
STRATEGY_AGGRESSIVE = 3   # Maps to StrategyActionEncoder action 4
STRATEGY_NAMES = ["UPGRADE", "FLUSH_CHASE", "STRAIGHT_CHASE", "AGGRESSIVE"]


class StateDiscretizer:
    """
    State discretization with draw indicators for hand improvement potential.

    Same as q_learning_agent.py - provides consistent state representation.
    """

    def __init__(self, config: Dict[str, Any], target_score: int):
        self.target_score = target_score
        self.chips_bins = np.array([
            0.0, 0.20, 0.40, 0.60, 0.75, 0.85, 0.90, 0.95, 1.00, 1.10, 1.25
        ])

    def _has_flush_draw(self, obs: Dict) -> int:
        if obs.get('has_flush', False):
            return 0
        return 1 if obs.get('flush_potential', False) else 0

    def _get_straight_draw_type(self, obs: Dict) -> int:
        if obs.get('has_straight', False):
            return 0
        ranks = sorted(set(obs['card_ranks']))
        if len(ranks) < 4:
            return 0
        for i in range(len(ranks) - 3):
            if ranks[i+3] - ranks[i] == 3:
                if ranks[i] > 0 and ranks[i+3] < 12:
                    return 2  # Open-ended
        for i in range(len(ranks) - 3):
            span = ranks[i+3] - ranks[i]
            if span == 4:
                gaps = [ranks[j+1] - ranks[j] for j in range(i, i+3)]
                if gaps.count(2) == 1 and gaps.count(1) == 2:
                    return 1  # Gutshot
        return 0

    def _get_upgrade_potential(self, obs: Dict) -> int:
        hand_type = obs['best_hand_type']
        if hand_type >= 5:
            return 0  # Low - already strong
        elif hand_type in [1, 2, 4]:
            return 1  # Medium
        else:
            return 2  # High

    def discretize(self, obs: Dict) -> Tuple:
        plays_left = obs['plays_left']
        discards_left = obs['discards_left']
        chips = obs['chips'][0]
        chips_progress = chips / self.target_score
        chips_bucket = np.digitize(chips_progress, self.chips_bins)
        best_hand_type = obs['best_hand_type']
        best_hand_score = obs['best_hand_score'][0]
        chips_needed = self.target_score - chips
        can_win_this_play = int(best_hand_score >= chips_needed)
        flush_draw = self._has_flush_draw(obs)
        straight_draw = self._get_straight_draw_type(obs)
        upgrade_potential = self._get_upgrade_potential(obs)

        return (
            plays_left,
            discards_left,
            chips_bucket,
            best_hand_type,
            can_win_this_play,
            flush_draw,
            straight_draw,
            upgrade_potential
        )


class HierarchicalQLearningAgent:
    """
    Hierarchical Q-Learning agent with two decision levels.

    Level 1 (Q_policy): Decides PLAY vs DISCARD
    Level 2 (Q_strategy): Decides which discard strategy (only when discarding)
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
        self.env = env
        self.state_discretizer = state_discretizer
        self.action_encoder = StrategyActionEncoder(env)

        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.default_q = default_q_value

        # Two Q-tables for hierarchical decision making
        # Q_policy: State -> {POLICY_PLAY: q, POLICY_DISCARD: q}
        self.q_policy = defaultdict(lambda: defaultdict(lambda: self.default_q))

        # Q_strategy: State -> {STRATEGY_UPGRADE: q, STRATEGY_FLUSH: q, ...}
        self.q_strategy = defaultdict(lambda: defaultdict(lambda: self.default_q))

        # Track last decisions for update
        self._last_policy_choice = None
        self._last_strategy_choice = None

        # Statistics
        self.total_updates = 0
        self.policy_updates = 0
        self.strategy_updates = 0

    def _get_valid_discard_strategies(self, obs: Dict) -> List[int]:
        """
        Get valid discard strategy indices (0-3) based on observation.

        UPGRADE and AGGRESSIVE are always valid when discards are available.
        FLUSH_CHASE requires flush_potential and not already having flush.
        STRAIGHT_CHASE requires straight_potential and not already having straight.
        """
        valid = [STRATEGY_UPGRADE, STRATEGY_AGGRESSIVE]

        if obs.get('flush_potential', 0) == 1 and obs.get('best_hand_type', 0) < 5:
            valid.append(STRATEGY_FLUSH)
        if obs.get('straight_potential', 0) == 1 and obs.get('best_hand_type', 0) < 4:
            valid.append(STRATEGY_STRAIGHT)

        return valid

    def _can_win_now(self, obs: Dict) -> bool:
        """Check if current best hand can reach target score."""
        best_score = obs['best_hand_score'][0]
        chips_needed = self.state_discretizer.target_score - obs['chips'][0]
        return best_score >= chips_needed

    def choose_action(self, obs: Dict, explore: bool = True) -> Dict:
        """
        Choose action using hierarchical epsilon-greedy policy.

        1. Q_policy decides: PLAY or DISCARD?
        2. If DISCARD, Q_strategy decides which strategy
        """
        state = self.state_discretizer.discretize(obs)
        plays_left = obs['plays_left']
        discards_left = obs['discards_left']

        # =====================================================================
        # LEVEL 1: Policy decision (PLAY vs DISCARD)
        # =====================================================================

        if explore and np.random.random() < self.epsilon:
            # Random policy choice
            if plays_left > 0 and discards_left > 0:
                policy_choice = np.random.choice([POLICY_PLAY, POLICY_DISCARD])
            elif plays_left > 0:
                policy_choice = POLICY_PLAY
            else:
                policy_choice = POLICY_DISCARD
        else:
            # Greedy policy choice
            q_play = self.q_policy[state].get(POLICY_PLAY, self.default_q)
            q_discard = self.q_policy[state].get(POLICY_DISCARD, self.default_q)

            # Only consider valid options
            if plays_left == 0:
                policy_choice = POLICY_DISCARD
            elif discards_left == 0:
                policy_choice = POLICY_PLAY
            else:
                policy_choice = POLICY_PLAY if q_play >= q_discard else POLICY_DISCARD

        self._last_policy_choice = policy_choice

        # =====================================================================
        # LEVEL 2: Execute policy choice
        # =====================================================================

        if policy_choice == POLICY_PLAY:
            # PLAY: Always use PLAY_BEST_HAND (action index 0)
            self._last_strategy_choice = None
            action = self.action_encoder.decode_to_env_action(0, obs)  # PLAY_BEST_HAND
            return action

        # DISCARD: Use Q_strategy to pick strategy
        valid_strategies = self._get_valid_discard_strategies(obs)

        if explore and np.random.random() < self.epsilon:
            # Random strategy choice
            strategy_choice = np.random.choice(valid_strategies)
        else:
            # Greedy strategy choice
            q_values = {s: self.q_strategy[state].get(s, self.default_q)
                       for s in valid_strategies}
            strategy_choice = max(q_values.items(), key=lambda x: x[1])[0]

        self._last_strategy_choice = strategy_choice

        # Map strategy choice to action encoder index
        # STRATEGY_UPGRADE=0 -> action 1 (DISCARD_UPGRADE)
        # STRATEGY_FLUSH=1 -> action 2 (DISCARD_FLUSH_CHASE)
        # etc.
        action_idx = strategy_choice + 1
        action = self.action_encoder.decode_to_env_action(action_idx, obs)

        return action

    def get_last_action_description(self) -> str:
        """Get description of last action for logging."""
        if self._last_policy_choice == POLICY_PLAY:
            return "PLAY -> PLAY_BEST_HAND"
        elif self._last_strategy_choice is not None:
            return f"DISCARD -> {STRATEGY_NAMES[self._last_strategy_choice]}"
        return "UNKNOWN"

    def update(self, obs: Dict, action: Dict, reward: float, next_obs: Dict, done: bool):
        """
        Update both Q-tables using standard Q-learning.

        Q_policy is always updated (we always make a PLAY/DISCARD decision).
        Q_strategy is only updated when we chose DISCARD.
        """
        state = self.state_discretizer.discretize(obs)
        next_state = self.state_discretizer.discretize(next_obs)

        # =====================================================================
        # Update Q_policy (PLAY vs DISCARD decision)
        # =====================================================================
        policy_choice = self._last_policy_choice
        if policy_choice is not None:
            current_q = self.q_policy[state].get(policy_choice, self.default_q)

            if done:
                target_q = reward
            else:
                # Max over valid policy actions in next state
                next_plays = next_obs['plays_left']
                next_discards = next_obs['discards_left']

                next_q_values = []
                if next_plays > 0:
                    next_q_values.append(self.q_policy[next_state].get(POLICY_PLAY, self.default_q))
                if next_discards > 0:
                    next_q_values.append(self.q_policy[next_state].get(POLICY_DISCARD, self.default_q))

                max_next_q = max(next_q_values) if next_q_values else self.default_q
                target_q = reward + self.gamma * max_next_q

            new_q = current_q + self.alpha * (target_q - current_q)
            self.q_policy[state][policy_choice] = new_q
            self.policy_updates += 1

        # =====================================================================
        # Update Q_strategy (only if we chose DISCARD)
        # =====================================================================
        strategy_choice = self._last_strategy_choice
        if strategy_choice is not None:
            current_q = self.q_strategy[state].get(strategy_choice, self.default_q)

            if done:
                target_q = reward
            else:
                # Max over valid strategies in next state
                valid_next_strategies = self._get_valid_discard_strategies(next_obs)
                if valid_next_strategies and next_obs['discards_left'] > 0:
                    next_q_values = [self.q_strategy[next_state].get(s, self.default_q)
                                    for s in valid_next_strategies]
                    max_next_q = max(next_q_values)
                else:
                    max_next_q = self.default_q
                target_q = reward + self.gamma * max_next_q

            new_q = current_q + self.alpha * (target_q - current_q)
            self.q_strategy[state][strategy_choice] = new_q
            self.strategy_updates += 1

        self.total_updates += 1

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_q_table_sizes(self) -> Tuple[int, int]:
        """Return sizes of both Q-tables."""
        return len(self.q_policy), len(self.q_strategy)


def train(
    agent: HierarchicalQLearningAgent,
    env: BalatroBatchedSimEnv,
    config: Dict,
    start_episode: int = 0
) -> Dict[str, List]:
    """Train hierarchical Q-learning agent."""
    num_episodes = config['training']['num_episodes']
    eval_freq = config['training']['eval_frequency']
    eval_episodes = config['training']['eval_episodes']
    checkpoint_freq = config['training']['checkpoint_frequency']
    verbose = config['logging']['verbose']
    log_freq = config['logging']['log_frequency']

    train_stats = {
        'episodes': [],
        'win_rates': [],
        'avg_rewards': [],
        'epsilons': []
    }

    # Evaluate random baseline
    if config['baseline']['compare_to_random'] and start_episode == 0:
        print("\nEvaluating random strategy baseline...")
        baseline_episodes = config['baseline']['random_episodes']
        baseline_win_rate, baseline_avg_reward = evaluate_random_strategy_baseline(
            env, baseline_episodes, seed=42
        )
        print(f"Random Baseline: Win Rate = {baseline_win_rate:.1%}, "
              f"Avg Reward = {baseline_avg_reward:.1f}")

    print(f"\nStarting hierarchical Q-learning training from episode {start_episode}...")
    print(f"Target episodes: {num_episodes}")
    print(f"Initial epsilon: {agent.epsilon:.3f}\n")

    for episode in range(start_episode, num_episodes):
        obs, _ = env.reset(seed=episode)
        episode_reward = 0.0
        steps = 0
        done = False
        max_steps = 100

        while not done and steps < max_steps:
            action = agent.choose_action(obs, explore=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            episode_reward += reward
            obs = next_obs
            steps += 1

        if steps >= max_steps and not done:
            print(f"\n[WARNING] Episode {episode+1} hit max steps limit")

        agent.decay_epsilon()

        if verbose and (episode + 1) % log_freq == 0:
            win_status = "WIN" if info.get('win', False) else "LOSS"
            policy_size, strategy_size = agent.get_q_table_sizes()
            print(f"Episode {episode + 1:5d}: Reward = {episode_reward:7.1f}, "
                  f"Steps = {steps:2d}, {win_status}, "
                  f"Policy={policy_size}, Strategy={strategy_size}, "
                  f"Epsilon = {agent.epsilon:.3f}")

        if (episode + 1) % eval_freq == 0:
            print(f"\nEvaluating agent...")
            win_rate, avg_reward = evaluate(agent, env, eval_episodes, seed=episode + 10000)

            train_stats['episodes'].append(episode + 1)
            train_stats['win_rates'].append(win_rate)
            train_stats['avg_rewards'].append(avg_reward)
            train_stats['epsilons'].append(agent.epsilon)

            policy_size, strategy_size = agent.get_q_table_sizes()
            print(f"\n{'=' * 60}")
            print(f"Evaluation at Episode {episode + 1}")
            print(f"{'=' * 60}")
            print(f"  Win Rate:       {win_rate:5.1%}")
            print(f"  Avg Reward:     {avg_reward:7.1f}")
            print(f"  Q_policy size:  {policy_size} states")
            print(f"  Q_strategy size:{strategy_size} states")
            print(f"  Epsilon:        {agent.epsilon:.3f}")
            print(f"{'=' * 60}\n")

        if (episode + 1) % checkpoint_freq == 0:
            # Use script directory as base for model path (consistent regardless of cwd)
            script_dir = Path(__file__).parent
            model_dir = script_dir / config['model']['save_dir']
            model_prefix = "hierarchical_q"
            filename = create_model_filename(model_prefix, episode + 1)
            filepath = model_dir / filename

            metadata = {
                'episode': episode + 1,
                'epsilon': agent.epsilon,
                'total_updates': agent.total_updates,
                'policy_updates': agent.policy_updates,
                'strategy_updates': agent.strategy_updates,
                'config': config
            }

            # Convert to regular dicts for pickling
            q_policy_dict = {state: dict(actions) for state, actions in agent.q_policy.items()}
            q_strategy_dict = {state: dict(actions) for state, actions in agent.q_strategy.items()}

            checkpoint = {
                'q_policy': q_policy_dict,
                'q_strategy': q_strategy_dict,
                'metadata': metadata
            }

            model_dir.mkdir(parents=True, exist_ok=True)
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint, f)

            print(f"Checkpoint saved to {filepath}")
            print(f"  Q_policy size: {len(q_policy_dict)} states")
            print(f"  Q_strategy size: {len(q_strategy_dict)} states")

    print("\nTraining complete!")
    policy_size, strategy_size = agent.get_q_table_sizes()
    print(f"Final Q_policy size: {policy_size} states")
    print(f"Final Q_strategy size: {strategy_size} states")
    print(f"Policy updates: {agent.policy_updates}")
    print(f"Strategy updates: {agent.strategy_updates}")

    return train_stats


def evaluate(
    agent: HierarchicalQLearningAgent,
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
                # Add hierarchical decision info
                decision = agent.get_last_action_description()
                if viz_mode == 'compact':
                    visualizer.visualize_step_compact(obs, action, next_obs, reward, info, decision=decision)
                else:
                    visualizer.visualize_step(obs, action, next_obs, reward, info, decision=decision)

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


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Q-Learning for Balatro Poker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--config", type=str, default="q_learning_config.yaml")
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-score", type=int, default=300)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--viz-mode", choices=["full", "compact"], default="full")

    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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
    env = BalatroBatchedSimEnv(target_score=target_score, reward_config_path=reward_config)

    # Create state discretizer
    state_discretizer = StateDiscretizer(config['state_discretization'], target_score)

    # Create or load agent
    if args.load_model:
        # Resolve model path relative to script directory
        script_dir = Path(__file__).parent
        model_path = Path(args.load_model)
        if not model_path.is_absolute():
            model_path = script_dir / model_path

        print(f"Loading model from {model_path}...")
        import pickle
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)

        metadata = checkpoint['metadata']

        agent = HierarchicalQLearningAgent(
            env, state_discretizer,
            learning_rate=config['learning']['alpha'],
            discount_factor=config['learning']['gamma'],
            epsilon_start=metadata.get('epsilon', config['learning']['epsilon_start']),
            epsilon_end=config['learning']['epsilon_end'],
            epsilon_decay=config['learning']['epsilon_decay'],
            default_q_value=config['q_table']['default_value']
        )

        # Load Q-tables
        for state, actions in checkpoint['q_policy'].items():
            agent.q_policy[state] = defaultdict(lambda: agent.default_q, actions)
        for state, actions in checkpoint['q_strategy'].items():
            agent.q_strategy[state] = defaultdict(lambda: agent.default_q, actions)

        agent.total_updates = metadata.get('total_updates', 0)
        agent.policy_updates = metadata.get('policy_updates', 0)
        agent.strategy_updates = metadata.get('strategy_updates', 0)

        start_episode = metadata.get('episode', 0)
        print(f"  Q_policy size: {len(agent.q_policy)} states")
        print(f"  Q_strategy size: {len(agent.q_strategy)} states")
        print(f"  Episode: {start_episode}")
    else:
        agent = HierarchicalQLearningAgent(
            env, state_discretizer,
            learning_rate=config['learning']['alpha'],
            discount_factor=config['learning']['gamma'],
            epsilon_start=config['learning']['epsilon_start'],
            epsilon_end=config['learning']['epsilon_end'],
            epsilon_decay=config['learning']['epsilon_decay'],
            default_q_value=config['q_table']['default_value']
        )
        start_episode = 0

    print("=" * 60)
    print("HIERARCHICAL Q-LEARNING AGENT FOR BALATRO POKER")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Target Score: {target_score}")
    print()
    print("Decision Hierarchy:")
    print("  Level 1 (Q_policy):   PLAY vs DISCARD")
    print("  Level 2 (Q_strategy): UPGRADE, FLUSH_CHASE, STRAIGHT_CHASE, AGGRESSIVE")
    print()
    print(f"Learning Rate: {agent.alpha}")
    print(f"Discount Factor: {agent.gamma}")
    print(f"Epsilon: {agent.epsilon:.3f}")
    policy_size, strategy_size = agent.get_q_table_sizes()
    print(f"Q_policy size: {policy_size} states")
    print(f"Q_strategy size: {strategy_size} states")
    print("=" * 60)

    if args.mode == "train":
        train_stats = train(agent, env, config, start_episode)

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

        if args.plot and train_stats['episodes']:
            script_dir = Path(__file__).parent
            plot_path = script_dir / config['model']['save_dir'] / "hierarchical_training_curves.png"
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


if __name__ == "__main__":
    main()
