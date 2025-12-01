"""
Utility functions for Q-learning agent.

Includes plotting, checkpointing, and baseline comparison.
"""

import pickle
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt


def save_checkpoint(q_table: Dict, metadata: Dict, filepath: str):
    """
    Save Q-table and training metadata to file.

    Args:
        q_table: Q-table dictionary {state_hash: {action_idx: q_value}}
        metadata: Training metadata (episode, epsilon, stats, etc.)
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'q_table': q_table,
        'metadata': metadata
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)

    print(f"Checkpoint saved to {filepath}")
    print(f"  Q-table size: {len(q_table)} states")
    print(f"  Episode: {metadata.get('episode', 'unknown')}")


def load_checkpoint(filepath: str) -> Tuple[Dict, Dict]:
    """
    Load Q-table and metadata from checkpoint file.

    Args:
        filepath: Path to checkpoint file

    Returns:
        (q_table, metadata) tuple
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)

    q_table = checkpoint['q_table']
    metadata = checkpoint['metadata']

    print(f"Checkpoint loaded from {filepath}")
    print(f"  Q-table size: {len(q_table)} states")
    print(f"  Episode: {metadata.get('episode', 'unknown')}")

    return q_table, metadata


def plot_training_curves(
    episodes: List[int],
    win_rates: List[float],
    avg_rewards: List[float],
    epsilons: List[float],
    save_path: str = None
):
    """
    Plot training progress curves.

    Args:
        episodes: List of episode numbers
        win_rates: List of win rates at each evaluation
        avg_rewards: List of average rewards at each evaluation
        epsilons: List of epsilon values
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Win rate plot
    axes[0].plot(episodes, win_rates, 'b-', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Win Rate')
    axes[0].set_title('Win Rate Over Training')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Average reward plot
    axes[1].plot(episodes, avg_rewards, 'g-', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Average Reward')
    axes[1].set_title('Average Reward Over Training')
    axes[1].grid(True, alpha=0.3)

    # Epsilon decay plot
    axes[2].plot(episodes, epsilons, 'r-', linewidth=2)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Epsilon')
    axes[2].set_title('Exploration Rate (Epsilon) Decay')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    plt.show()


def compare_to_baseline(
    agent_win_rate: float,
    agent_avg_reward: float,
    baseline_win_rate: float,
    baseline_avg_reward: float
):
    """
    Print comparison between agent and baseline performance.

    Args:
        agent_win_rate: Agent's win rate
        agent_avg_reward: Agent's average reward
        baseline_win_rate: Baseline's win rate
        baseline_avg_reward: Baseline's average reward
    """
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON: Q-Learning vs Random Baseline")
    print("=" * 60)

    print(f"\n{'Metric':<30} {'Q-Learning':>12} {'Random':>12} {'Improvement':>12}")
    print("-" * 60)

    # Win rate comparison
    win_rate_improvement = ((agent_win_rate - baseline_win_rate) / max(baseline_win_rate, 0.001)) * 100
    print(f"{'Win Rate':<30} {agent_win_rate:>11.1%} {baseline_win_rate:>11.1%} {win_rate_improvement:>11.1f}%")

    # Reward comparison
    reward_improvement = ((agent_avg_reward - baseline_avg_reward) / abs(baseline_avg_reward + 1e-6)) * 100
    print(f"{'Average Reward':<30} {agent_avg_reward:>12.1f} {baseline_avg_reward:>12.1f} {reward_improvement:>11.1f}%")

    print("=" * 60)

    # Verdict
    if agent_win_rate > baseline_win_rate * 1.1:  # 10% better
        print("[+] Q-Learning agent significantly outperforms random baseline!")
    elif agent_win_rate > baseline_win_rate:
        print("[~] Q-Learning agent slightly outperforms random baseline")
    else:
        print("[-] Q-Learning agent needs more training (currently underperforming)")


def print_q_table_stats(q_table: Dict):
    """
    Print statistics about the Q-table.

    Args:
        q_table: Q-table dictionary
    """
    if not q_table:
        print("Q-table is empty")
        return

    # Count total state-action pairs
    total_pairs = sum(len(actions) for actions in q_table.values())

    # Get Q-value statistics
    all_q_values = []
    for state_actions in q_table.values():
        all_q_values.extend(state_actions.values())

    if not all_q_values:
        print("No Q-values stored yet")
        return

    print("\n" + "=" * 50)
    print("Q-TABLE STATISTICS")
    print("=" * 50)
    print(f"  Total states: {len(q_table)}")
    print(f"  Total state-action pairs: {total_pairs}")
    print(f"  Avg actions per state: {total_pairs / len(q_table):.1f}")
    print(f"\nQ-Value Statistics:")
    print(f"  Min: {np.min(all_q_values):.2f}")
    print(f"  Max: {np.max(all_q_values):.2f}")
    print(f"  Mean: {np.mean(all_q_values):.2f}")
    print(f"  Std: {np.std(all_q_values):.2f}")
    print("=" * 50)


def create_model_filename(prefix: str, episode: int, extension: str = "pkl") -> str:
    """
    Create standardized model filename.

    Args:
        prefix: Model name prefix
        episode: Episode number
        extension: File extension

    Returns:
        Filename string
    """
    return f"{prefix}_ep{episode:06d}.{extension}"


def evaluate_random_baseline(env, num_episodes: int, seed: int = 42) -> Tuple[float, float]:
    """
    Evaluate random baseline agent.

    Args:
        env: Balatro environment
        num_episodes: Number of episodes to evaluate
        seed: Random seed

    Returns:
        (win_rate, avg_reward) tuple
    """
    import numpy as np

    np_random = np.random.RandomState(seed)
    wins = 0
    total_reward = 0.0

    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed + i)
        episode_reward = 0.0
        done = False
        steps = 0
        max_steps = 100  # Safety limit

        while not done and steps < max_steps:
            # Generate random action
            action_type = np_random.choice([0, 1])  # PLAY or DISCARD
            num_cards = np_random.randint(1, 9) if action_type == 1 else np_random.randint(1, 6)
            card_mask = np.zeros(8, dtype=np.int8)
            selected = np_random.choice(8, size=min(num_cards, 8), replace=False)
            card_mask[selected] = 1

            action = {
                'type': action_type,
                'card_mask': card_mask
            }

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1

            if done and info.get('win', False):
                wins += 1

        total_reward += episode_reward

    win_rate = wins / num_episodes
    avg_reward = total_reward / num_episodes

    return win_rate, avg_reward
