#!/usr/bin/env python3
"""
Seed replay tool for determinism validation.

Usage:
    python replay.py <target_score> <seed> <action1> <action2> ...

Example:
    python replay.py 300 12345 0 2 0 1

Action codes:
    0 = PLAY_BEST
    1 = PLAY_PAIR
    2 = DISCARD_NON_PAIRED
    3 = DISCARD_LOWEST_3
    4 = PLAY_HIGHEST_3
    5 = RANDOM_PLAY
    6 = RANDOM_DISCARD
"""

import sys
import os
import hashlib

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python', 'balatro_env'))
import _balatro_core as core


def replay_episode(target_score, seed, actions):
    """
    Replay an episode and return trajectory hash.

    Returns:
        dict with keys: trajectory_hash, rewards, done, win, final_obs
    """
    sim = core.Simulator()
    sim.reset(target_score, seed)

    result = sim.step_batch(actions)

    # Create trajectory hash
    trajectory_data = (
        str(list(result.final_obs)) +
        str(list(result.rewards)) +
        str(result.done) +
        str(result.win)
    )
    trajectory_hash = hashlib.sha256(trajectory_data.encode()).hexdigest()

    return {
        'trajectory_hash': trajectory_hash,
        'rewards': list(result.rewards),
        'done': result.done,
        'win': result.win,
        'final_obs': list(result.final_obs),
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    try:
        target_score = int(sys.argv[1])
        seed = int(sys.argv[2])
        actions = [int(a) for a in sys.argv[3:]]
    except ValueError:
        print("Error: target_score, seed, and actions must be integers")
        sys.exit(1)

    print(f"Replaying episode:")
    print(f"  Target Score: {target_score}")
    print(f"  Seed: {seed}")
    print(f"  Actions: {actions}")
    print()

    result = replay_episode(target_score, seed, actions)

    print("Results:")
    print(f"  Trajectory Hash: {result['trajectory_hash']}")
    print(f"  Rewards: {result['rewards']}")
    print(f"  Done: {result['done']}")
    print(f"  Win: {result['win']}")
    print(f"  Final Observation: {result['final_obs']}")
    print()

    # Verify determinism by running twice
    print("Verifying determinism...")
    result2 = replay_episode(target_score, seed, actions)

    if result['trajectory_hash'] == result2['trajectory_hash']:
        print("[PASS] Determinism verified - same hash on replay")
    else:
        print("[FAIL] Determinism broken - different hashes!")
        sys.exit(1)


if __name__ == "__main__":
    main()
