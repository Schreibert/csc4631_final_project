#!/usr/bin/env python3
"""
Naive Heuristic Agent for Balatro Poker Environment.

Uses simple card-counting heuristics based on basic game knowledge:
- Prioritize cards of the same rank (pairs, trips, quads)
- Prioritize cards of the same suit (flush potential)
- Play the best grouping available
- Discard cards that don't contribute to any pattern

This is a simpler, more "human-like" approach compared to the
strategy-based heuristic that uses the C++ hand evaluator.
"""

import sys
import os
import argparse
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from balatro_env import BalatroBatchedSimEnv
from agent_visualizer import AgentVisualizer


class NaiveHeuristicAgent:
    """
    Naive heuristic agent using simple card-counting logic.

    Strategy:
    1. Count cards by rank -> find pairs, trips, quads
    2. Count cards by suit -> find flush potential
    3. Play the strongest pattern available
    4. Discard cards that don't fit any pattern
    """

    # Hand type names for logging
    HAND_NAMES = [
        "High Card", "Pair", "Two Pair", "Three of a Kind",
        "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"
    ]

    def __init__(self, env):
        """
        Initialize naive heuristic agent.

        Args:
            env: BalatroBatchedSimEnv instance
        """
        self.env = env
        self._last_action_name = None
        self._last_reason = None

    def _analyze_hand(self, obs: Dict) -> Dict:
        """
        Analyze the current hand for patterns.

        Returns dict with:
        - rank_counts: Counter of ranks
        - suit_counts: Counter of suits
        - rank_groups: {rank: [indices]} for cards of same rank
        - suit_groups: {suit: [indices]} for cards of same suit
        - best_rank_group: (rank, indices) of largest rank group
        - best_suit_group: (suit, indices) of largest suit group
        """
        ranks = list(obs['card_ranks'])
        suits = list(obs['card_suits'])

        # Count by rank and suit
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        # Group indices by rank
        rank_groups = {}
        for i, rank in enumerate(ranks):
            if rank not in rank_groups:
                rank_groups[rank] = []
            rank_groups[rank].append(i)

        # Group indices by suit
        suit_groups = {}
        for i, suit in enumerate(suits):
            if suit not in suit_groups:
                suit_groups[suit] = []
            suit_groups[suit].append(i)

        # Find best rank group (most cards of same rank, prefer higher rank)
        best_rank = None
        best_rank_indices = []
        for rank, indices in rank_groups.items():
            if len(indices) > len(best_rank_indices) or \
               (len(indices) == len(best_rank_indices) and rank > (best_rank or -1)):
                best_rank = rank
                best_rank_indices = indices

        # Find best suit group (most cards of same suit)
        best_suit = None
        best_suit_indices = []
        for suit, indices in suit_groups.items():
            if len(indices) > len(best_suit_indices):
                best_suit = suit
                best_suit_indices = indices

        return {
            'rank_counts': rank_counts,
            'suit_counts': suit_counts,
            'rank_groups': rank_groups,
            'suit_groups': suit_groups,
            'best_rank_group': (best_rank, best_rank_indices),
            'best_suit_group': (best_suit, best_suit_indices),
            'ranks': ranks,
            'suits': suits
        }

    def _find_two_pair_indices(self, rank_groups: Dict) -> Optional[List[int]]:
        """Find indices for two pair if available."""
        pairs = [(rank, indices) for rank, indices in rank_groups.items() if len(indices) >= 2]
        if len(pairs) >= 2:
            # Sort by rank descending, take top 2 pairs
            pairs.sort(key=lambda x: x[0], reverse=True)
            indices = pairs[0][1][:2] + pairs[1][1][:2]
            return indices
        return None

    def _find_full_house_indices(self, rank_groups: Dict) -> Optional[List[int]]:
        """Find indices for full house if available."""
        trips = [(rank, indices) for rank, indices in rank_groups.items() if len(indices) >= 3]
        pairs = [(rank, indices) for rank, indices in rank_groups.items() if len(indices) >= 2]

        if trips and len(pairs) >= 2:
            # Have trips and at least one other pair
            trip_rank, trip_indices = max(trips, key=lambda x: x[0])
            # Find a pair that's not the trips
            for pair_rank, pair_indices in sorted(pairs, key=lambda x: x[0], reverse=True):
                if pair_rank != trip_rank:
                    return trip_indices[:3] + pair_indices[:2]
        return None

    def _select_play_cards(self, analysis: Dict) -> Tuple[List[int], str]:
        """
        Select which cards to play based on analysis.

        Returns:
            (indices to play, reason string)
        """
        rank_groups = analysis['rank_groups']
        suit_groups = analysis['suit_groups']
        best_rank, best_rank_indices = analysis['best_rank_group']
        best_suit, best_suit_indices = analysis['best_suit_group']
        ranks = analysis['ranks']

        # Check for Four of a Kind (4 cards same rank)
        if len(best_rank_indices) >= 4:
            return best_rank_indices[:4], "four_of_kind"

        # Check for Full House (3 + 2 of different ranks)
        full_house = self._find_full_house_indices(rank_groups)
        if full_house:
            return full_house, "full_house"

        # Check for Flush (5+ cards same suit)
        if len(best_suit_indices) >= 5:
            # Sort by rank descending, take top 5
            suited_with_ranks = [(i, ranks[i]) for i in best_suit_indices]
            suited_with_ranks.sort(key=lambda x: x[1], reverse=True)
            return [i for i, _ in suited_with_ranks[:5]], "flush"

        # Check for Three of a Kind
        if len(best_rank_indices) >= 3:
            return best_rank_indices[:3], "three_of_kind"

        # Check for Two Pair
        two_pair = self._find_two_pair_indices(rank_groups)
        if two_pair:
            return two_pair, "two_pair"

        # Check for Pair
        if len(best_rank_indices) >= 2:
            return best_rank_indices[:2], "pair"

        # High Card - play highest ranked card
        ranked_indices = sorted(range(8), key=lambda i: ranks[i], reverse=True)
        return ranked_indices[:1], "high_card"

    def _select_discard_cards(self, analysis: Dict, obs: Dict) -> Tuple[List[int], str]:
        """
        Select which cards to discard based on analysis.

        Strategy:
        - Keep cards that contribute to rank patterns (pairs, trips)
        - Keep cards that contribute to suit patterns (flush potential)
        - Discard the rest, prioritizing low-rank cards
        """
        rank_groups = analysis['rank_groups']
        suit_groups = analysis['suit_groups']
        ranks = analysis['ranks']

        # Find cards worth keeping
        keep_indices = set()

        # Keep all cards in rank groups of 2+
        for rank, indices in rank_groups.items():
            if len(indices) >= 2:
                keep_indices.update(indices)

        # Keep all cards in suit groups of 4+ (flush potential)
        for suit, indices in suit_groups.items():
            if len(indices) >= 4:
                keep_indices.update(indices)

        # Cards to potentially discard
        discard_candidates = [i for i in range(8) if i not in keep_indices]

        if not discard_candidates:
            # All cards are "valuable" - discard lowest ranked cards
            ranked_indices = sorted(range(8), key=lambda i: ranks[i])
            discard_candidates = ranked_indices[:3]
            reason = "discard_lowest"
        else:
            # Sort candidates by rank (discard lowest first)
            discard_candidates.sort(key=lambda i: ranks[i])
            reason = "discard_non_contributing"

        # Discard up to 5 cards (game limit)
        num_to_discard = min(5, len(discard_candidates))
        return discard_candidates[:num_to_discard], reason

    def choose_action(self, obs: Dict, np_random=None) -> Dict:
        """
        Choose action based on naive heuristics.

        Args:
            obs: Observation dictionary
            np_random: Unused, kept for interface compatibility

        Returns:
            Action dict with 'type' and 'card_mask'
        """
        plays_left = obs['plays_left']
        discards_left = obs['discards_left']

        # Analyze current hand
        analysis = self._analyze_hand(obs)
        best_rank, best_rank_indices = analysis['best_rank_group']
        best_suit, best_suit_indices = analysis['best_suit_group']

        # Decision logic
        card_mask = np.zeros(8, dtype=np.int8)

        # If no plays left, must discard
        if plays_left == 0:
            if discards_left > 0:
                indices, reason = self._select_discard_cards(analysis, obs)
                for i in indices:
                    card_mask[i] = 1
                self._last_action_name = "DISCARD"
                self._last_reason = reason
                return {'type': 1, 'card_mask': card_mask}
            return None

        # Decide: Play or Discard?
        # Play if we have a strong hand (trips+) or no discards left
        # Play if we have a pair and only 1 play left
        # Otherwise consider discarding to improve

        has_trips_or_better = len(best_rank_indices) >= 3
        has_flush = len(best_suit_indices) >= 5
        has_pair = len(best_rank_indices) >= 2
        has_flush_draw = len(best_suit_indices) >= 4

        # Strong hand -> Play
        if has_trips_or_better or has_flush:
            indices, reason = self._select_play_cards(analysis)
            for i in indices:
                card_mask[i] = 1
            self._last_action_name = "PLAY"
            self._last_reason = reason
            return {'type': 0, 'card_mask': card_mask}

        # Last play, no discards -> Play whatever we have
        if plays_left == 1 and discards_left == 0:
            indices, reason = self._select_play_cards(analysis)
            for i in indices:
                card_mask[i] = 1
            self._last_action_name = "PLAY"
            self._last_reason = f"{reason}_forced"
            return {'type': 0, 'card_mask': card_mask}

        # Have discards and weak hand -> Try to improve
        if discards_left > 0:
            # If we have flush draw, discard non-suited cards
            if has_flush_draw and not has_pair:
                best_suit = analysis['best_suit_group'][0]
                discard_indices = [i for i in range(8) if analysis['suits'][i] != best_suit]
                discard_indices = discard_indices[:5]  # Max 5 discards
                if discard_indices:
                    for i in discard_indices:
                        card_mask[i] = 1
                    self._last_action_name = "DISCARD"
                    self._last_reason = "chase_flush"
                    return {'type': 1, 'card_mask': card_mask}

            # If we only have high card or weak pair, discard non-contributing
            if not has_pair or (has_pair and discards_left >= 2):
                indices, reason = self._select_discard_cards(analysis, obs)
                if indices:
                    for i in indices:
                        card_mask[i] = 1
                    self._last_action_name = "DISCARD"
                    self._last_reason = reason
                    return {'type': 1, 'card_mask': card_mask}

        # Default: Play best hand
        indices, reason = self._select_play_cards(analysis)
        for i in indices:
            card_mask[i] = 1
        self._last_action_name = "PLAY"
        self._last_reason = reason
        return {'type': 0, 'card_mask': card_mask}

    def get_last_action_name(self) -> str:
        """Get name of last action for logging."""
        return self._last_action_name or "UNKNOWN"

    def get_last_reason(self) -> str:
        """Get reason for last decision."""
        return self._last_reason or "unknown"


def run_episode(env, agent, seed, verbose=False, visualize=False, viz_mode='full'):
    """Run one episode with naive heuristic agent."""
    obs, info = env.reset(seed=seed)

    visualizer = None
    if visualize:
        visualizer = AgentVisualizer()
        visualizer.reset_episode()

    total_reward = 0
    step = 0

    while step < 100:
        if verbose and not visualize:
            plays_left = obs['plays_left']
            discards_left = obs['discards_left']
            chips = obs['chips'][0] if hasattr(obs['chips'], '__len__') else obs['chips']
            print(f"\n--- Step {step} ---")
            print(f"State: plays={plays_left}, discards={discards_left}, chips={chips}/{env.target_score}")
            print(f"Cards: ranks={list(obs['card_ranks'])}, suits={list(obs['card_suits'])}")

        action = agent.choose_action(obs)

        if action is None:
            if verbose or visualize:
                print("No valid actions available!")
            break

        if verbose and not visualize:
            action_name = agent.get_last_action_name()
            reason = agent.get_last_reason()
            action_type = "PLAY" if action['type'] == 0 else "DISCARD"
            selected = [i for i, m in enumerate(action['card_mask']) if m]
            print(f"Action: {action_type} ({reason})")
            print(f"Selected cards: {selected}")

        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if visualize:
            action_name = agent.get_last_action_name()
            reason = agent.get_last_reason()
            decision = f"{action_name} ({reason})"
            if viz_mode == 'compact':
                visualizer.visualize_step_compact(obs, action, next_obs, reward, info, decision=decision)
            else:
                visualizer.visualize_step(obs, action, next_obs, reward, info, decision=decision)

        obs = next_obs

        if terminated:
            if verbose and not visualize:
                print(f"\n{'WIN!' if info['win'] else 'LOSS'}")
                print(f"Final chips: {info['chips']}")
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
        description="Naive heuristic agent for Balatro poker (card-counting based)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["run", "eval"],
        default="run",
        help="Mode: 'run' for batch episodes, 'eval' for visualized evaluation"
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
        "--visualize",
        action="store_true",
        help="Show agent decisions with card details"
    )
    parser.add_argument(
        "--viz-mode",
        choices=["full", "compact"],
        default="full",
        help="Visualization mode"
    )

    args = parser.parse_args()

    if args.mode == "eval":
        args.visualize = True

    print("=" * 60)
    print("NAIVE HEURISTIC AGENT")
    print("(Card-counting based decisions)")
    print("=" * 60)
    print(f"Running {args.episodes} episodes...")
    print(f"Target score: {args.target_score}")
    print(f"Starting seed: {args.seed}")
    print()
    print("Strategy:")
    print("  - Count cards by rank (pairs, trips, quads)")
    print("  - Count cards by suit (flush potential)")
    print("  - Play strongest pattern available")
    print("  - Discard non-contributing cards")
    if args.visualize:
        print(f"\nVisualization: {args.viz_mode} mode")
    print()

    env = BalatroBatchedSimEnv(target_score=args.target_score)
    agent = NaiveHeuristicAgent(env)

    if args.visualize and args.episodes > 0:
        visualizer = AgentVisualizer()
        visualizer.print_episode_header(1, args.target_score, seed=args.seed)
        result = run_episode(env, agent, seed=args.seed, visualize=True, viz_mode=args.viz_mode)
        visualizer.print_episode_summary(result)

        if args.episodes == 1:
            return

        print(f"\nRunning {args.episodes - 1} more episodes...\n")
        starting_episode = 1
    elif args.verbose:
        result = run_episode(env, agent, seed=args.seed, verbose=True)
        print("\n" + "=" * 60 + "\n")
        starting_episode = 0
    else:
        starting_episode = 0

    wins = 0
    total_steps = 0
    total_rewards = 0
    total_chips = 0

    for i in range(starting_episode, args.episodes):
        result = run_episode(env, agent, seed=args.seed + i, verbose=False, visualize=False)

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
    print(f"  Win rate:       {wins/args.episodes:5.1%}")
    print(f"  Average steps:  {total_steps/args.episodes:4.1f}")
    print(f"  Average reward: {total_rewards/args.episodes:7.1f}")
    print(f"  Average chips:  {total_chips/args.episodes:5.1f}")
    print(f"{'=' * 60}")


def evaluate_naive_heuristic_baseline(env, num_episodes: int, seed: int = 42):
    """
    Evaluate naive heuristic baseline for comparison.

    Returns:
        Tuple of (win_rate, avg_reward)
    """
    agent = NaiveHeuristicAgent(env)
    wins = 0
    total_reward = 0.0

    for i in range(num_episodes):
        result = run_episode(env, agent, seed=seed + i, verbose=False, visualize=False)

        if result['win']:
            wins += 1
        total_reward += result['total_reward']

    return wins / num_episodes, total_reward / num_episodes


if __name__ == "__main__":
    main()
