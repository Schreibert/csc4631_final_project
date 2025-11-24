#!/usr/bin/env python3
"""
Demonstration of custom reward configurations.

This example shows how to:
1. Use the default YAML reward configuration
2. Create custom reward configurations
3. Compare different reward schemes
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
        num_cards = np_random.randint(1, 6)
    else:  # DISCARD
        num_cards = np_random.randint(1, 9)

    # Create random card mask
    card_mask = np.zeros(8, dtype=np.int8)
    selected_indices = np_random.choice(8, size=num_cards, replace=False)
    card_mask[selected_indices] = 1

    return {
        'type': action_type,
        'card_mask': card_mask
    }


def run_episode(env, seed):
    """Run one episode with random actions"""
    obs, info = env.reset(seed=seed)
    np_random = np.random.RandomState(seed)

    total_reward = 0
    step = 0

    while step < 100:  # Max 100 steps as safety
        action = generate_random_action(obs, np_random)
        if action is None:
            break

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if terminated:
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


def benchmark_config(env, num_episodes=100, desc="Config"):
    """Run multiple episodes and collect statistics"""
    wins = 0
    total_steps = 0
    total_rewards = 0
    total_chips = 0

    for i in range(num_episodes):
        result = run_episode(env, seed=i)

        if result['win']:
            wins += 1

        total_steps += result['steps']
        total_rewards += result['total_reward']
        total_chips += result['final_chips']

    return {
        'description': desc,
        'win_rate': wins / num_episodes,
        'avg_steps': total_steps / num_episodes,
        'avg_reward': total_rewards / num_episodes,
        'avg_chips': total_chips / num_episodes,
    }


def main():
    target_score = 300
    num_episodes = 50  # Use fewer episodes for demo

    print("=" * 70)
    print("Custom Reward Configuration Demo")
    print("=" * 70)
    print(f"Target score: {target_score}")
    print(f"Episodes per config: {num_episodes}\n")

    # =============================================================================
    # 1. Default Configuration (from rewards_config.yaml)
    # =============================================================================
    print("\n" + "=" * 70)
    print("1. DEFAULT CONFIGURATION (Conservative Efficiency)")
    print("=" * 70)

    env_default = BalatroBatchedSimEnv(target_score=target_score)
    print(env_default.reward_shaper.get_config_summary())
    print("\nRunning benchmark...")
    results_default = benchmark_config(env_default, num_episodes, "Default")

    # =============================================================================
    # 2. Aggressive Configuration (maximize score, high hand bonuses)
    # =============================================================================
    print("\n" + "=" * 70)
    print("2. AGGRESSIVE CONFIGURATION (High Score Focus)")
    print("=" * 70)

    aggressive_config = {
        'outcome': {
            'win_bonus': 500,           # Lower win bonus (focus on score itself)
            'loss_penalty': 1000,       # Higher loss penalty
        },
        'efficiency': {
            'play_conservation_bonus': 0,  # No conservation bonus (use all plays)
            'step_penalty': 0.5,        # Lower step penalty (take time to maximize score)
        },
        'progress': {
            'chip_gain_scale': 2.0,     # DOUBLE the chip reward scaling
            'chip_normalization': 50,   # Less normalization = bigger chip rewards
            'target_threshold_bonuses': [
                {'threshold': 0.5, 'bonus': 50},
                {'threshold': 0.75, 'bonus': 100},
                {'threshold': 0.90, 'bonus': 150},
            ],
        },
        'hand_quality': {
            'enabled': True,
            'bonuses': {
                'high_card': 0,
                'pair': 5,
                'two_pair': 10,
                'three_of_a_kind': 15,
                'straight': 25,
                'flush': 30,
                'full_house': 40,
                'four_of_a_kind': 50,
                'straight_flush': 100,  # Huge bonus for rare hands
            },
        },
        'penalties': {
            'invalid_action': 50,
            'desperate_play': 5,        # Smaller desperate play penalty
            'desperate_threshold': 0.3,
        },
        'advanced': {
            'safety_margin_bonus': {
                'enabled': True,
                'per_chip_over_target': 0.5,  # Reward overkill
                'max_bonus': 200,
            },
            'exploration': {
                'enabled': False,
                'action_diversity_bonus': 0,
            },
        },
        'version': 'custom-aggressive',
        'description': 'Aggressive high-score focused rewards',
    }

    env_aggressive = BalatroBatchedSimEnv(
        target_score=target_score,
        reward_config=aggressive_config
    )
    print(env_aggressive.reward_shaper.get_config_summary())
    print("\nRunning benchmark...")
    results_aggressive = benchmark_config(env_aggressive, num_episodes, "Aggressive")

    # =============================================================================
    # 3. Sparse Configuration (only terminal rewards, minimal shaping)
    # =============================================================================
    print("\n" + "=" * 70)
    print("3. SPARSE CONFIGURATION (Win/Loss Only)")
    print("=" * 70)

    sparse_config = {
        'outcome': {
            'win_bonus': 1000,
            'loss_penalty': 1000,       # Symmetric win/loss
        },
        'efficiency': {
            'play_conservation_bonus': 0,
            'step_penalty': 0,          # No step penalty
        },
        'progress': {
            'chip_gain_scale': 0.1,     # Minimal chip rewards
            'chip_normalization': 1000, # Heavy normalization
            'target_threshold_bonuses': [],  # No threshold bonuses
        },
        'hand_quality': {
            'enabled': False,           # No hand bonuses
            'bonuses': {},
        },
        'penalties': {
            'invalid_action': 10,
            'desperate_play': 0,
            'desperate_threshold': 0.5,
        },
        'advanced': {
            'safety_margin_bonus': {
                'enabled': False,
                'per_chip_over_target': 0,
                'max_bonus': 0,
            },
            'exploration': {
                'enabled': False,
                'action_diversity_bonus': 0,
            },
        },
        'version': 'custom-sparse',
        'description': 'Sparse rewards (terminal only)',
    }

    env_sparse = BalatroBatchedSimEnv(
        target_score=target_score,
        reward_config=sparse_config
    )
    print(env_sparse.reward_shaper.get_config_summary())
    print("\nRunning benchmark...")
    results_sparse = benchmark_config(env_sparse, num_episodes, "Sparse")

    # =============================================================================
    # 4. Compare Results
    # =============================================================================
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    all_results = [results_default, results_aggressive, results_sparse]

    print(f"\n{'Configuration':<25} {'Win Rate':>10} {'Avg Steps':>10} {'Avg Reward':>12} {'Avg Chips':>10}")
    print("-" * 70)

    for result in all_results:
        print(f"{result['description']:<25} "
              f"{result['win_rate']:>9.1%} "
              f"{result['avg_steps']:>10.1f} "
              f"{result['avg_reward']:>12.1f} "
              f"{result['avg_chips']:>10.1f}")

    print("\n" + "=" * 70)
    print("Key Observations:")
    print("  - Default: Balanced rewards with efficiency focus")
    print("  - Aggressive: Higher total rewards due to chip scaling & hand bonuses")
    print("  - Sparse: Simple binary feedback (hardest to learn from)")
    print("\nNote: All configs use the same random agent, so win rates should be similar.")
    print("The key difference is in HOW rewards are distributed during episodes.")
    print("=" * 70)


if __name__ == "__main__":
    main()
