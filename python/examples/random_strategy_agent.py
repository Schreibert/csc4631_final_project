#!/usr/bin/env python3
"""
Random Strategy Agent for Balatro Poker Environment.

Uses the same strategy-based action space as the Q-learning agent
for fair comparison. Randomly selects from valid strategies.

Strategies (5 actions):
- 0: PLAY_BEST_HAND - Play optimal 5-card hand
- 1: DISCARD_UPGRADE - Keep best-hand cards, discard others
- 2: DISCARD_FLUSH_CHASE - Chase flush by discarding non-flush cards
- 3: DISCARD_STRAIGHT_CHASE - Chase straight by discarding non-consecutive
- 4: DISCARD_AGGRESSIVE - Discard 5 lowest cards for complete reset
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


class RandomStrategyAgent:
    """
    Random agent that selects from strategy-based actions.

    Uses the same StrategyActionEncoder as the Q-learning agent,
    but chooses randomly from valid strategies.
    """

    def __init__(self, env):
        """
        Initialize random strategy agent.

        Args:
            env: BalatroBatchedSimEnv instance
        """
        self.env = env
        self.action_encoder = StrategyActionEncoder(env)
        self._last_strategy_idx = None

    def choose_action(self, obs, np_random):
        """
        Choose a random valid strategy action.

        Args:
            obs: Current observation
            np_random: NumPy random state for reproducibility

        Returns:
            Environment action dict with 'type' and 'card_mask'
        """
        # Get valid strategies for current state
        valid_strategies = self.action_encoder.get_valid_actions(obs)

        # Randomly select a strategy
        strategy_idx = np_random.choice(valid_strategies)
        self._last_strategy_idx = strategy_idx

        # Convert strategy to environment action
        action = self.action_encoder.decode_to_env_action(strategy_idx, obs)

        return action

    def get_last_strategy_name(self):
        """Get name of last chosen strategy for logging."""
        if self._last_strategy_idx is not None:
            return self.action_encoder.get_action_name(self._last_strategy_idx)
        return "UNKNOWN"


def run_episode(env, agent, seed, verbose=False, visualize=False, viz_mode='full'):
    """Run one episode with random strategy selection.

    Args:
        env: Environment instance
        agent: RandomStrategyAgent instance
        seed: Random seed for episode
        verbose: Legacy verbose mode (prints raw state)
        visualize: If True, use AgentVisualizer to show decisions
        viz_mode: 'full' for detailed view, 'compact' for one-line summaries
    """
    obs, info = env.reset(seed=seed)

    # Create RNG for this episode
    np_random = np.random.RandomState(seed)

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
            chips = obs['chips'][0]
            print(f"\n--- Step {step} ---")
            print(f"State: plays={plays_left}, discards={discards_left}, chips={chips}/{env.target_score}")
            print(f"Best hand type: {obs['best_hand_type']}")

        # Choose random strategy action
        action = agent.choose_action(obs, np_random)

        if action is None:
            if verbose or visualize:
                print("No valid actions available!")
            break

        if verbose and not visualize:
            strategy_name = agent.get_last_strategy_name()
            action_type = "PLAY" if action['type'] == 0 else "DISCARD"
            selected = [i for i, m in enumerate(action['card_mask']) if m]
            print(f"Strategy: {strategy_name}")
            print(f"Action: {action_type} cards {selected}")

        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Visualize decision with strategy name
        if visualize:
            strategy_name = agent.get_last_strategy_name()
            if viz_mode == 'compact':
                # Print strategy name before compact view
                print(f"  Strategy: {strategy_name}")
                visualizer.visualize_step_compact(obs, action, next_obs, reward, info)
            else:  # full mode
                print(f"\n  Strategy Selected: {strategy_name}")
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
        description="Random strategy agent for Balatro poker (uses same action space as Q-learning)",
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
    print("RANDOM STRATEGY AGENT")
    print("(Same action space as Q-Learning)")
    print("=" * 60)
    print(f"Running {num_episodes} episodes with random strategy selection...")
    print(f"Target score: {target_score}")
    print(f"Starting seed: {starting_seed}")
    print(f"Action Space: 5 strategies")
    print(f"  - PLAY_BEST_HAND, DISCARD_UPGRADE, DISCARD_FLUSH_CHASE,")
    print(f"    DISCARD_STRAIGHT_CHASE, DISCARD_AGGRESSIVE")
    if visualize:
        print(f"Visualization: {viz_mode} mode")
    print()

    # Create environment
    env = BalatroBatchedSimEnv(
        target_score=target_score,
        reward_config_path=args.reward_config
    )

    # Create agent
    agent = RandomStrategyAgent(env)

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


def evaluate_random_strategy_baseline(env, num_episodes: int, seed: int = 42):
    """
    Evaluate random strategy baseline for comparison with Q-learning.

    Args:
        env: BalatroBatchedSimEnv instance
        num_episodes: Number of episodes to evaluate
        seed: Starting random seed

    Returns:
        Tuple of (win_rate, avg_reward)
    """
    agent = RandomStrategyAgent(env)
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
