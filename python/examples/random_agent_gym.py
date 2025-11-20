#!/usr/bin/env python3
"""
Random agent using Gymnasium wrapper.

Demonstrates using the BalatroBatchedSimEnv with proper seeding
for reproducible random episodes.
"""

import sys
import os
import numpy as np

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from balatro_env import BalatroBatchedSimEnv


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
        # Discard 1-8 cards randomly
        num_cards = np_random.randint(1, 9)

    # Create random card mask
    card_mask = np.zeros(8, dtype=np.int8)
    selected_indices = np_random.choice(8, size=num_cards, replace=False)
    card_mask[selected_indices] = 1

    return {
        'type': action_type,
        'card_mask': card_mask
    }


def run_episode(env, seed, verbose=False):
    """Run one episode with random actions"""
    obs, info = env.reset(seed=seed)

    # Create RNG for this episode
    np_random = np.random.RandomState(seed)

    total_reward = 0
    step = 0

    while step < 100:  # Max 100 steps as safety
        if verbose:
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
            if verbose:
                print("No valid actions available!")
            break

        if verbose:
            action_type = "PLAY" if action['type'] == 0 else "DISCARD"
            selected = [i for i, m in enumerate(action['card_mask']) if m]
            print(f"Action: {action_type} cards {selected}")

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if verbose and reward > 0:
            print(f"Reward: {reward:.1f}")

        if terminated:
            if verbose:
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
    target_score = 300
    num_episodes = 100
    verbose = False  # Set to True to see detailed output

    print("=" * 60)
    print("Random Agent - Gymnasium Wrapper")
    print("=" * 60)
    print(f"Running {num_episodes} episodes with random actions...")
    print(f"Target score: {target_score}\n")

    # Create environment
    env = BalatroBatchedSimEnv(target_score=target_score)

    if verbose:
        # Run one verbose episode first
        print("\n=== Verbose Episode Example ===")
        result = run_episode(env, seed=42, verbose=True)
        print("\n" + "=" * 60 + "\n")

    wins = 0
    total_steps = 0
    total_rewards = 0
    total_chips = 0

    for i in range(num_episodes):
        result = run_episode(env, seed=i, verbose=False)

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
