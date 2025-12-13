#!/usr/bin/env python3
"""
Basic Heuristic Agent for Balatro Poker Environment.

Uses a "Greedy with Discard Threshold" strategy:
    - Play best hand immediately if it can win or is strong enough
    - Otherwise, try to upgrade via discards if resources available
    - Considers hand type, score potential, and remaining resources

Decision Priority (evaluated in order):
    1. Can win now?              -> Play best hand (immediate victory)
    2. No plays left?            -> Must discard (if available)
    3. Strong hand (3-of-kind+)? -> Play it (high value)
    4. Two Pair + few discards?  -> Play it (not worth chasing)
    5. Last play remaining?      -> Play best hand (forced)
    6. Flush potential + discards? -> Chase flush
    7. Straight potential + discards? -> Chase straight
    8. Weak hand + discards?     -> Discard to upgrade
    9. Fallback                  -> Play best hand
"""

import sys
import os
import argparse
import numpy as np

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from balatro_env import BalatroBatchedSimEnv
from agent_visualizer import AgentVisualizer
from strategy_action_encoder import StrategyActionEncoder


class BasicHeuristicAgent:
    """
    Heuristic-based agent using greedy play with discard thresholds.

    Decision Priority:
    1. Can win now? -> Play best hand
    2. No plays left? -> Must discard (if available)
    3. Strong hand (3-of-kind+)? -> Play it
    4. Two Pair with few discards? -> Play it
    5. Flush potential + discards? -> Chase flush
    6. Straight potential + discards? -> Chase straight
    7. Weak hand + discards? -> Discard to upgrade
    8. Fallback -> Play best hand
    """

    # Hand type thresholds (from HandType enum)
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8

    def __init__(self, env):
        """
        Initialize heuristic agent.

        Args:
            env: BalatroBatchedSimEnv instance
        """
        self.env = env
        self.action_encoder = StrategyActionEncoder(env)
        self._last_strategy_idx = None
        self._last_reason = None

    def choose_action(self, obs, np_random=None):
        """
        Choose action based on heuristic rules.

        Args:
            obs: Current observation dict
            np_random: NumPy random state (unused, kept for interface compatibility)

        Returns:
            Environment action dict with 'type' and 'card_mask'
        """
        plays_left = obs['plays_left']
        discards_left = obs['discards_left']
        chips = obs['chips'][0] if isinstance(obs['chips'], np.ndarray) else obs['chips']
        best_hand_type = obs['best_hand_type']
        best_hand_score = obs['best_hand_score']
        flush_potential = obs.get('flush_potential', 0)
        straight_potential = obs.get('straight_potential', 0)
        chips_to_target = obs['chips_to_target']

        # 1. Can win now? -> Play immediately
        if best_hand_score >= chips_to_target:
            self._last_strategy_idx = StrategyActionEncoder.PLAY_BEST_HAND
            self._last_reason = "can_win_now"
            return self.action_encoder.decode_to_env_action(
                StrategyActionEncoder.PLAY_BEST_HAND, obs
            )

        # 2. No plays left? -> Must discard if possible
        if plays_left == 0:
            if discards_left > 0:
                self._last_strategy_idx = StrategyActionEncoder.DISCARD_UPGRADE
                self._last_reason = "no_plays_must_discard"
                return self.action_encoder.decode_to_env_action(
                    StrategyActionEncoder.DISCARD_UPGRADE, obs
                )
            # No plays and no discards - shouldn't happen, but fallback
            return None

        # 3. Strong hand (Three of a Kind or better)? -> Play it
        if best_hand_type >= self.THREE_OF_A_KIND:
            self._last_strategy_idx = StrategyActionEncoder.PLAY_BEST_HAND
            self._last_reason = "strong_hand"
            return self.action_encoder.decode_to_env_action(
                StrategyActionEncoder.PLAY_BEST_HAND, obs
            )

        # 4. Two Pair with few discards remaining? -> Play it
        if best_hand_type == self.TWO_PAIR and discards_left <= 1:
            self._last_strategy_idx = StrategyActionEncoder.PLAY_BEST_HAND
            self._last_reason = "two_pair_low_discards"
            return self.action_encoder.decode_to_env_action(
                StrategyActionEncoder.PLAY_BEST_HAND, obs
            )

        # 5. Last play remaining? -> Play best hand (can't afford to discard)
        if plays_left == 1 and discards_left == 0:
            self._last_strategy_idx = StrategyActionEncoder.PLAY_BEST_HAND
            self._last_reason = "last_play_no_discards"
            return self.action_encoder.decode_to_env_action(
                StrategyActionEncoder.PLAY_BEST_HAND, obs
            )

        # Now consider discarding to improve hand
        if discards_left > 0:
            # 6. Flush potential? -> Chase flush (high value)
            if flush_potential and best_hand_type < self.FLUSH:
                self._last_strategy_idx = StrategyActionEncoder.DISCARD_FLUSH_CHASE
                self._last_reason = "chase_flush"
                return self.action_encoder.decode_to_env_action(
                    StrategyActionEncoder.DISCARD_FLUSH_CHASE, obs
                )

            # 7. Straight potential? -> Chase straight
            if straight_potential and best_hand_type < self.STRAIGHT:
                self._last_strategy_idx = StrategyActionEncoder.DISCARD_STRAIGHT_CHASE
                self._last_reason = "chase_straight"
                return self.action_encoder.decode_to_env_action(
                    StrategyActionEncoder.DISCARD_STRAIGHT_CHASE, obs
                )

            # 8. Weak hand (Pair or High Card) with discards? -> Try to upgrade
            if best_hand_type <= self.PAIR:
                self._last_strategy_idx = StrategyActionEncoder.DISCARD_UPGRADE
                self._last_reason = "weak_hand_upgrade"
                return self.action_encoder.decode_to_env_action(
                    StrategyActionEncoder.DISCARD_UPGRADE, obs
                )

            # 9. Two Pair with discards? -> Try for full house
            if best_hand_type == self.TWO_PAIR:
                self._last_strategy_idx = StrategyActionEncoder.DISCARD_UPGRADE
                self._last_reason = "two_pair_upgrade"
                return self.action_encoder.decode_to_env_action(
                    StrategyActionEncoder.DISCARD_UPGRADE, obs
                )

        # 10. Fallback: Play best hand
        self._last_strategy_idx = StrategyActionEncoder.PLAY_BEST_HAND
        self._last_reason = "fallback"
        return self.action_encoder.decode_to_env_action(
            StrategyActionEncoder.PLAY_BEST_HAND, obs
        )

    def get_last_strategy_name(self):
        """Get name of last chosen strategy for logging."""
        if self._last_strategy_idx is not None:
            return self.action_encoder.get_action_name(self._last_strategy_idx)
        return "UNKNOWN"

    def get_last_reason(self):
        """Get the reason for the last decision."""
        return self._last_reason or "unknown"


def run_episode(env, agent, seed, verbose=False, visualize=False, viz_mode='full'):
    """Run one episode with heuristic decision making.

    Args:
        env: Environment instance
        agent: BasicHeuristicAgent instance
        seed: Random seed for episode
        verbose: Legacy verbose mode (prints raw state)
        visualize: If True, use AgentVisualizer to show decisions
        viz_mode: 'full' for detailed view, 'compact' for one-line summaries
    """
    obs, info = env.reset(seed=seed)

    # Create visualizer if requested
    visualizer = None
    if visualize:
        visualizer = AgentVisualizer()
        visualizer.reset_episode()

    total_reward = 0
    step = 0

    while step < 100:  # Max 100 steps as safety
        if verbose and not visualize:
            plays_left = obs['plays_left']
            discards_left = obs['discards_left']
            chips = obs['chips'][0] if isinstance(obs['chips'], np.ndarray) else obs['chips']
            print(f"\n--- Step {step} ---")
            print(f"State: plays={plays_left}, discards={discards_left}, chips={chips}/{env.target_score}")
            print(f"Best hand type: {obs['best_hand_type']} (score: {obs['best_hand_score']})")

        # Choose heuristic action
        action = agent.choose_action(obs)

        if action is None:
            if verbose or visualize:
                print("No valid actions available!")
            break

        if verbose and not visualize:
            strategy_name = agent.get_last_strategy_name()
            reason = agent.get_last_reason()
            action_type = "PLAY" if action['type'] == 0 else "DISCARD"
            selected = [i for i, m in enumerate(action['card_mask']) if m]
            print(f"Strategy: {strategy_name} (reason: {reason})")
            print(f"Action: {action_type} cards {selected}")

        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Visualize decision with strategy name and reason
        if visualize:
            strategy_name = agent.get_last_strategy_name()
            reason = agent.get_last_reason()
            if viz_mode == 'compact':
                print(f"  Strategy: {strategy_name} ({reason})")
                visualizer.visualize_step_compact(obs, action, next_obs, reward, info)
            else:  # full mode
                print(f"\n  Strategy: {strategy_name}")
                print(f"  Reason: {reason}")
                visualizer.visualize_step(obs, action, next_obs, reward, info)

        obs = next_obs

        if verbose and reward > 0 and not visualize:
            print(f"Reward: {reward:.1f}")

        if terminated:
            if verbose and not visualize:
                print(f"\n{'WIN!' if info['win'] else 'LOSS'}")
                print(f"Final chips: {info['chips']}")
                print(f"Steps: {step}")
            return {
                'win': info['win'],
                'steps': step,
                'total_reward': total_reward,
                'final_chips': info['chips'],
            }

    return {
        'win': False,
        'steps': step,
        'total_reward': total_reward,
        'final_chips': info.get('chips', 0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Basic heuristic agent for Balatro poker (greedy with discard threshold)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["run", "eval"],
        default="run",
        help="Mode: 'run' for batch episodes with stats, 'eval' for visualized evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Starting seed for episode generation"
    )
    parser.add_argument(
        "--target-score",
        type=int,
        default=300,
        help="Target score to beat the blind"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed episode information"
    )
    parser.add_argument(
        "--reward-config",
        type=str,
        default=None,
        help="Path to custom reward configuration YAML file"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show agent decisions with card details"
    )
    parser.add_argument(
        "--viz-mode",
        choices=["full", "compact"],
        default="full",
        help="Visualization mode: 'full' for detailed view, 'compact' for one-line summaries"
    )

    args = parser.parse_args()

    # --mode eval implies --visualize
    if args.mode == "eval":
        args.visualize = True

    target_score = args.target_score
    num_episodes = args.episodes
    verbose = args.verbose
    visualize = args.visualize
    viz_mode = args.viz_mode
    starting_seed = args.seed

    print("=" * 60)
    print("BASIC HEURISTIC AGENT")
    print("(Greedy with Discard Threshold)")
    print("=" * 60)
    print(f"Running {num_episodes} episodes with heuristic strategy...")
    print(f"Target score: {target_score}")
    print(f"Starting seed: {starting_seed}")
    print()
    print("Decision Priority:")
    print("  1. Can win now? -> Play best hand")
    print("  2. Strong hand (3-of-kind+)? -> Play it")
    print("  3. Flush/Straight potential? -> Chase it")
    print("  4. Weak hand + discards? -> Upgrade")
    print("  5. Fallback -> Play best hand")
    if visualize:
        print(f"\nVisualization: {viz_mode} mode")
    print()

    # Create environment
    env = BalatroBatchedSimEnv(
        target_score=target_score,
        reward_config_path=args.reward_config
    )

    # Create agent
    agent = BasicHeuristicAgent(env)

    # Print reward configuration being used
    if not visualize:
        print(env.reward_shaper.get_config_summary())
        print()

    # If visualizing, show first episode in detail
    if visualize and num_episodes > 0:
        visualizer = AgentVisualizer()
        visualizer.print_episode_header(1, target_score, seed=starting_seed)
        result = run_episode(env, agent, seed=starting_seed, visualize=True, viz_mode=viz_mode)
        visualizer.print_episode_summary(result)

        if num_episodes == 1:
            return

        print(f"\nRunning {num_episodes - 1} more episodes...\n")
        starting_episode = 1
    elif verbose:
        print("\n=== Verbose Episode Example ===")
        result = run_episode(env, agent, seed=starting_seed, verbose=True)
        print("\n" + "=" * 60 + "\n")
        starting_episode = 0
    else:
        starting_episode = 0

    wins = 0
    total_steps = 0
    total_rewards = 0
    total_chips = 0

    for i in range(starting_episode, num_episodes):
        result = run_episode(env, agent, seed=starting_seed + i, verbose=False, visualize=False)

        if result['win']:
            wins += 1

        total_steps += result['steps']
        total_rewards += result['total_reward']
        total_chips += result['final_chips']

        if (i + 1) % 10 == 0:
            print(f"Episode {i+1:3d}: " +
                  f"Win rate = {wins/(i+1):5.1%}, " +
                  f"Avg steps = {total_steps/(i+1):4.1f}, " +
                  f"Avg reward = {total_rewards/(i+1):7.1f}, " +
                  f"Avg chips = {total_chips/(i+1):5.1f}")

    print(f"\n{'=' * 60}")
    print("Final Results:")
    print(f"  Win rate:       {wins/num_episodes:5.1%}")
    print(f"  Average steps:  {total_steps/num_episodes:4.1f}")
    print(f"  Average reward: {total_rewards/num_episodes:7.1f}")
    print(f"  Average chips:  {total_chips/num_episodes:5.1f}")
    print(f"{'=' * 60}")


def evaluate_heuristic_baseline(env, num_episodes: int, seed: int = 42):
    """
    Evaluate heuristic baseline for comparison with Q-learning.

    Args:
        env: BalatroBatchedSimEnv instance
        num_episodes: Number of episodes to evaluate
        seed: Starting random seed

    Returns:
        Tuple of (win_rate, avg_reward)
    """
    agent = BasicHeuristicAgent(env)
    wins = 0
    total_reward = 0.0

    for i in range(num_episodes):
        result = run_episode(env, agent, seed=seed + i, verbose=False, visualize=False)

        if result['win']:
            wins += 1
        total_reward += result['total_reward']

    win_rate = wins / num_episodes
    avg_reward = total_reward / num_episodes

    return win_rate, avg_reward


if __name__ == "__main__":
    main()
