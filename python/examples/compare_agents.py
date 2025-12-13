#!/usr/bin/env python3
"""
Agent Performance Comparison for Balatro Poker Environment.

Compares performance of:
1. Basic Heuristic Agent (greedy with discard threshold)
2. Random Strategy Agent (random selection from 5 strategies)
3. Q-Learning Agent (hierarchical Q-learning, loaded from checkpoint)

Outputs:
- Console summary table
- Bar chart visualization (PNG)
"""

import sys
import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from balatro_env import BalatroBatchedSimEnv
from basic_heuristic import BasicHeuristicAgent, evaluate_heuristic_baseline
from random_strategy_agent import RandomStrategyAgent, evaluate_random_strategy_baseline
from hierarchical_q_learning_agent import (
    HierarchicalQLearningAgent, StateDiscretizer, evaluate
)


def load_q_learning_agent(env, checkpoint_path, target_score):
    """Load hierarchical Q-learning agent from checkpoint."""
    print(f"Loading Q-learning model from {checkpoint_path}...")

    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    metadata = checkpoint['metadata']
    config = metadata.get('config', {})

    # Create state discretizer
    state_config = config.get('state_discretization', {})
    state_discretizer = StateDiscretizer(state_config, target_score)

    # Create agent with default parameters
    agent = HierarchicalQLearningAgent(
        env, state_discretizer,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=0.0,  # No exploration during evaluation
        epsilon_end=0.0,
        epsilon_decay=1.0,
        default_q_value=0.0
    )

    # Load Q-tables
    for state, actions in checkpoint['q_policy'].items():
        agent.q_policy[state] = defaultdict(lambda: agent.default_q, actions)
    for state, actions in checkpoint['q_strategy'].items():
        agent.q_strategy[state] = defaultdict(lambda: agent.default_q, actions)

    episode = metadata.get('episode', 'unknown')
    print(f"  Loaded model trained for {episode} episodes")
    print(f"  Q_policy size: {len(agent.q_policy)} states")
    print(f"  Q_strategy size: {len(agent.q_strategy)} states")

    return agent


def evaluate_all_agents(env, q_agent, num_episodes, seed):
    """Evaluate all three agents and return results."""
    results = {}

    # 1. Evaluate Q-Learning Agent
    print(f"\nEvaluating Q-Learning Agent ({num_episodes} episodes)...")
    q_win_rate, q_avg_reward = evaluate(q_agent, env, num_episodes, seed=seed)
    results['Q-Learning'] = {
        'win_rate': q_win_rate,
        'avg_reward': q_avg_reward
    }
    print(f"  Win Rate: {q_win_rate:.1%}, Avg Reward: {q_avg_reward:.1f}")

    # 2. Evaluate Basic Heuristic Agent
    print(f"\nEvaluating Basic Heuristic Agent ({num_episodes} episodes)...")
    h_win_rate, h_avg_reward = evaluate_heuristic_baseline(env, num_episodes, seed=seed)
    results['Heuristic'] = {
        'win_rate': h_win_rate,
        'avg_reward': h_avg_reward
    }
    print(f"  Win Rate: {h_win_rate:.1%}, Avg Reward: {h_avg_reward:.1f}")

    # 3. Evaluate Random Strategy Agent
    print(f"\nEvaluating Random Strategy Agent ({num_episodes} episodes)...")
    r_win_rate, r_avg_reward = evaluate_random_strategy_baseline(env, num_episodes, seed=seed)
    results['Random'] = {
        'win_rate': r_win_rate,
        'avg_reward': r_avg_reward
    }
    print(f"  Win Rate: {r_win_rate:.1%}, Avg Reward: {r_avg_reward:.1f}")

    return results


def print_comparison_table(results):
    """Print comparison table to console."""
    print("\n" + "=" * 70)
    print("AGENT PERFORMANCE COMPARISON")
    print("=" * 70)

    # Get random baseline for improvement calculation
    random_win = results['Random']['win_rate']
    random_reward = results['Random']['avg_reward']

    print(f"\n{'Agent':<20} {'Win Rate':>12} {'Avg Reward':>12} {'vs Random':>15}")
    print("-" * 70)

    for agent_name in ['Q-Learning', 'Heuristic', 'Random']:
        win_rate = results[agent_name]['win_rate']
        avg_reward = results[agent_name]['avg_reward']

        if agent_name == 'Random':
            improvement = "baseline"
        else:
            if random_win > 0:
                win_improvement = ((win_rate - random_win) / random_win) * 100
                improvement = f"{win_improvement:+.1f}%"
            else:
                improvement = "N/A"

        print(f"{agent_name:<20} {win_rate:>11.1%} {avg_reward:>12.1f} {improvement:>15}")

    print("=" * 70)

    # Determine winner
    best_agent = max(results.keys(), key=lambda x: results[x]['win_rate'])
    print(f"\nBest performing agent: {best_agent} ({results[best_agent]['win_rate']:.1%} win rate)")


def plot_comparison(results, output_path, target_score, num_episodes):
    """Create bar chart comparison visualization."""
    agents = ['Random', 'Heuristic', 'Q-Learning']
    win_rates = [results[a]['win_rate'] * 100 for a in agents]
    avg_rewards = [results[a]['avg_reward'] for a in agents]

    # Colors
    colors = ['#e74c3c', '#f39c12', '#27ae60']  # Red, Orange, Green

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Win Rate Bar Chart
    bars1 = axes[0].bar(agents, win_rates, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Win Rate (%)', fontsize=12)
    axes[0].set_title('Win Rate Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars1, win_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Average Reward Bar Chart
    bars2 = axes[1].bar(agents, avg_rewards, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('Average Reward', fontsize=12)
    axes[1].set_title('Average Reward Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars2, avg_rewards):
        ypos = bar.get_height() + (max(avg_rewards) * 0.02 if val >= 0 else -max(avg_rewards) * 0.08)
        axes[1].text(bar.get_x() + bar.get_width()/2, ypos,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add subtitle with evaluation info
    fig.suptitle(f'Balatro Agent Performance Comparison\n(Target: {target_score}, Episodes: {num_episodes})',
                 fontsize=14, y=1.02)

    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nComparison chart saved to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare performance of Balatro poker agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per agent"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Starting random seed for evaluation"
    )
    parser.add_argument(
        "--target-score",
        type=int,
        default=300,
        help="Target score for the blind"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/hierarchical_q_ep030000.pkl",
        help="Path to Q-learning checkpoint (relative to script directory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/agent_comparison.png",
        help="Output path for comparison chart (relative to script directory)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the comparison plot"
    )

    args = parser.parse_args()

    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent
    checkpoint_path = script_dir / args.checkpoint
    output_path = script_dir / args.output

    # Check checkpoint exists
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("\nAvailable checkpoints:")
        models_dir = script_dir / "models"
        if models_dir.exists():
            checkpoints = sorted(models_dir.glob("hierarchical_q_*.pkl"))
            for cp in checkpoints[-5:]:  # Show last 5
                print(f"  {cp.name}")
        return 1

    print("=" * 70)
    print("BALATRO AGENT PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"Target Score: {args.target_score}")
    print(f"Evaluation Episodes: {args.episodes}")
    print(f"Random Seed: {args.seed}")
    print(f"Q-Learning Checkpoint: {checkpoint_path.name}")
    print("=" * 70)

    # Create environment
    env = BalatroBatchedSimEnv(target_score=args.target_score)

    # Load Q-learning agent
    q_agent = load_q_learning_agent(env, checkpoint_path, args.target_score)

    # Evaluate all agents
    results = evaluate_all_agents(env, q_agent, args.episodes, args.seed)

    # Print comparison table
    print_comparison_table(results)

    # Generate plot
    if not args.no_plot:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_comparison(results, output_path, args.target_score, args.episodes)

    return 0


if __name__ == "__main__":
    sys.exit(main())
