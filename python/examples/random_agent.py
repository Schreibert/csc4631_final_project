#!/usr/bin/env python3
"""
Random agent using Gymnasium wrapper.

Demonstrates using the BalatroBatchedSimEnv with proper seeding
for reproducible random episodes.
"""

import sys
import os
import argparse
import numpy as np

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from balatro_env import BalatroBatchedSimEnv
from agent_visualizer import AgentVisualizer


def generate_random_action(obs, np_random):
    """Generate a random valid action based on current observation"""
    plays_left = obs['plays_left']
    discards_left = obs['discards_left']

    # Randomly choose to play or discard
    if plays_left > 0 and discards_left > 0:
        action_type = np_random.choice([0, 1])  # 0=PLAY, 1=DISCARD
    elif plays_left > 0:
        action_type = 0  # PLAY
    elif discards_left > 0:
        action_type = 1  # DISCARD
    else:
        return None  # No valid actions

    # Generate random card selection
    if action_type == 0:  # PLAY
        # Play 1-5 cards randomly
        num_cards = np_random.randint(1, 6)
    else:  # DISCARD
        # Discard 1-5 cards randomly
        num_cards = np_random.randint(1, 6)

    # Create random card mask
    card_mask = np.zeros(8, dtype=np.int8)
    selected_indices = np_random.choice(8, size=num_cards, replace=False)
    card_mask[selected_indices] = 1

    return {
        'type': action_type,
        'card_mask': card_mask
    }


def run_episode(env, seed, verbose=False, visualize=False, viz_mode='full'):
    """Run one episode with random actions

    Args:
        env: Environment instance
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
            chips_to_target = obs['chips_to_target'][0]
            print(f"\n--- Step {step} ---")
            print(f"State: plays={plays_left}, discards={discards_left}, chips={chips}/{env.target_score}")
            print(f"Cards: ranks={obs['card_ranks']}, suits={obs['card_suits']}")

        # Generate random action
        action = generate_random_action(obs, np_random)

        if action is None:
            if verbose or visualize:
                print("No valid actions available!")
            break

        if verbose and not visualize:
            action_type = "PLAY" if action['type'] == 0 else "DISCARD"
            selected = [i for i, m in enumerate(action['card_mask']) if m]
            print(f"Action: {action_type} cards {selected}")

        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Visualize decision
        if visualize:
            if viz_mode == 'compact':
                visualizer.visualize_step_compact(obs, action, next_obs, reward, info)
            else:  # full mode
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
        description="Run random agent on Balatro poker environment",
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
        help="Starting seed for episode generation (episodes will use seed, seed+1, seed+2, ...)"
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
        help="Print detailed episode information (legacy mode)"
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
        help="Show agent decisions with card details (hand dealt, action, result)"
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
    print("Random Agent - Gymnasium Wrapper")
    print("=" * 60)
    print(f"Running {num_episodes} episodes with random actions...")
    print(f"Target score: {target_score}")
    print(f"Starting seed: {starting_seed}")
    if visualize:
        print(f"Visualization: {viz_mode} mode")
    print()

    # Create environment with YAML reward configuration
    # By default, uses rewards_config.yaml for reward shaping
    env = BalatroBatchedSimEnv(
        target_score=target_score,
        reward_config_path=args.reward_config
    )

    # Print reward configuration being used
    if not visualize:  # Skip config summary in visualize mode to reduce clutter
        print(env.reward_shaper.get_config_summary())
        print()

    # If visualizing, show first episode in detail
    if visualize and num_episodes > 0:
        visualizer = AgentVisualizer()
        visualizer.print_episode_header(1, target_score, seed=starting_seed)
        result = run_episode(env, seed=starting_seed, visualize=True, viz_mode=viz_mode)
        visualizer.print_episode_summary(result)

        # If only one episode requested, we're done
        if num_episodes == 1:
            return

        # Otherwise, continue with remaining episodes
        print(f"\nRunning {num_episodes - 1} more episodes...\n")
        starting_episode = 1
    elif verbose:
        # Run one verbose episode first
        print("\n=== Verbose Episode Example ===")
        result = run_episode(env, seed=starting_seed, verbose=True)
        print("\n" + "=" * 60 + "\n")
        starting_episode = 0
    else:
        starting_episode = 0

    wins = 0
    total_steps = 0
    total_rewards = 0
    total_chips = 0

    for i in range(starting_episode, num_episodes):
        result = run_episode(env, seed=starting_seed + i, verbose=False, visualize=False)

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


if __name__ == "__main__":
    main()
