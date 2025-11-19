#!/usr/bin/env python3
"""
Simple random agent example.

Demonstrates basic usage of the Balatro environment.
"""

import sys
import os
import random

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'balatro_env'))
import _balatro_core as core


def run_episode(target_score, seed):
    """Run one episode with random actions"""
    sim = core.Simulator()
    sim.reset(target_score, seed)

    total_reward = 0
    step = 0

    while step < 100:  # Max 100 steps as safety
        # Random action
        action = random.randint(0, core.NUM_ACTIONS - 1)

        result = sim.step_batch([action])
        total_reward += result.rewards[0]
        step += 1

        if result.done:
            return {
                'win': result.win,
                'steps': step,
                'total_reward': total_reward,
            }

    return {'win': False, 'steps': step, 'total_reward': total_reward}


def main():
    target_score = 300
    num_episodes = 100

    print(f"Running {num_episodes} episodes with random agent...")
    print(f"Target score: {target_score}\n")

    wins = 0
    total_steps = 0

    for i in range(num_episodes):
        result = run_episode(target_score, seed=i)

        if result['win']:
            wins += 1

        total_steps += result['steps']

        if (i + 1) % 10 == 0:
            print(f"Episode {i+1}: Win rate = {wins/(i+1):.1%}, Avg steps = {total_steps/(i+1):.1f}")

    print(f"\nFinal results:")
    print(f"  Win rate: {wins/num_episodes:.1%}")
    print(f"  Average steps: {total_steps/num_episodes:.1f}")


if __name__ == "__main__":
    main()
