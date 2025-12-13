#!/usr/bin/env python3
"""
Agent decision visualization utilities for Balatro poker simulator.

Provides the AgentVisualizer class for displaying agent decisions including:
    - Hand dealt (cards with ranks and suits)
    - Action chosen (play/discard with selected cards)
    - Result (points scored, hand type, scoring breakdown)
    - Episode summaries (win/loss, total chips, steps taken)

Features:
    - Two visualization modes: 'full' (detailed) and 'compact' (one-line)
    - Platform-aware: Unicode suit symbols on Linux/Mac, ASCII on Windows
    - Scoring formula breakdown showing base chips, rank sum, and multiplier
    - Hand type analysis by reverse-engineering from chip scores

Output Modes:
    Full mode example:
        ======================================================================
        STEP 1
        ======================================================================
        Resources: 4 plays, 3 discards remaining
        Progress:  0/300 chips (0.0%)

        Hand dealt: [K♠, K♦, K♥, A♣, A♦, 7♠, 3♦, 2♣]
        Best possible: Full House (368 chips)

        >>> ACTION: PLAY 5 card(s)
            Selected: [K♠, K♦, K♥, A♣, A♦]

        <<< RESULT: Full House
            Base: 40 chips × 4
            Rank sum: 10+10+10+11+11 = 52
            Total: (40 + 52) × 4 = 368 chips

        Reward: 1.25 | WIN!

    Compact mode example:
        Step  1: PLAY [K♠, K♦, K♥, A♣, A♦] -> +368 chips (total: 368) | WIN!
"""

import sys
import os
import platform
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from balatro_env import _balatro_core as core


# Card rank and suit display names
RANK_NAMES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUIT_NAMES = ['♣', '♦', '♥', '♠']  # Clubs, Diamonds, Hearts, Spades
SUIT_NAMES_ASCII = ['C', 'D', 'H', 'S']  # ASCII fallback

# Rank values for scoring (2-10 = face value, J/Q/K = 10, A = 11)
RANK_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]

HAND_TYPE_NAMES = [
    'High Card',
    'Pair',
    'Two Pair',
    'Three of a Kind',
    'Straight',
    'Flush',
    'Full House',
    'Four of a Kind',
    'Straight Flush'
]

# Base chips and multipliers for each hand type (from scoring.cpp)
# Format: (base_chips, base_mult)
HAND_SCORING = [
    (5, 1),    # HIGH_CARD
    (10, 2),   # PAIR
    (20, 2),   # TWO_PAIR
    (30, 3),   # THREE_OF_A_KIND
    (30, 4),   # STRAIGHT
    (35, 4),   # FLUSH
    (40, 4),   # FULL_HOUSE
    (60, 7),   # FOUR_OF_A_KIND
    (100, 8)   # STRAIGHT_FLUSH
]


class AgentVisualizer:
    """Visualizes agent decisions during episodes."""

    def __init__(self, use_ascii=None):
        """
        Initialize visualizer.

        Args:
            use_ascii: If True, use ASCII suit symbols (C,D,H,S) instead of Unicode.
                      If None (default), auto-detect based on platform/encoding.
        """
        # Auto-detect if not specified
        if use_ascii is None:
            # Use ASCII on Windows or if stdout encoding doesn't support Unicode
            use_ascii = (platform.system() == 'Windows' or
                        not hasattr(sys.stdout, 'encoding') or
                        sys.stdout.encoding.lower() not in ['utf-8', 'utf8'])

        self.use_ascii = use_ascii
        self.suit_names = SUIT_NAMES_ASCII if use_ascii else SUIT_NAMES
        self.step_count = 0

    def reset_episode(self):
        """Reset step counter for new episode."""
        self.step_count = 0

    def format_card(self, rank, suit):
        """
        Format a single card as a string.

        Args:
            rank: Card rank (0-12, where 0=2, 12=A)
            suit: Card suit (0-3, where 0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades)

        Returns:
            String like "A♠" or "10♥"
        """
        # Convert numpy int32 to Python int
        rank = int(rank)
        suit = int(suit)

        # Validate indices
        if rank < 0 or rank >= len(RANK_NAMES):
            return f"?{rank}?"
        if suit < 0 or suit >= len(self.suit_names):
            return f"?{suit}?"

        return f"{RANK_NAMES[rank]}{self.suit_names[suit]}"

    def format_hand(self, obs):
        """
        Format the current hand as a string.

        Args:
            obs: Observation dict with 'card_ranks' and 'card_suits'

        Returns:
            String like "[A♠, K♥, Q♦, J♣, 10♠, 9♥, 8♦, 7♣]"
        """
        cards = []
        for rank, suit in zip(obs['card_ranks'], obs['card_suits']):
            cards.append(self.format_card(rank, suit))
        return f"[{', '.join(cards)}]"

    def format_selected_cards(self, obs, card_mask):
        """
        Format only the selected cards from the hand.

        Args:
            obs: Observation dict with 'card_ranks' and 'card_suits'
            card_mask: Boolean array indicating which cards are selected

        Returns:
            String like "[A♠, K♥, Q♦]" (only selected cards)
        """
        cards = []
        for i, (rank, suit, selected) in enumerate(zip(obs['card_ranks'], obs['card_suits'], card_mask)):
            if selected:
                cards.append(self.format_card(rank, suit))
        return f"[{', '.join(cards)}]"

    def get_hand_type_name(self, hand_type_idx):
        """
        Get human-readable name for hand type.

        Args:
            hand_type_idx: HandType enum value (0-8)

        Returns:
            String like "Straight Flush" or "Pair"
        """
        if 0 <= hand_type_idx < len(HAND_TYPE_NAMES):
            return HAND_TYPE_NAMES[hand_type_idx]
        return f"Unknown({hand_type_idx})"

    def analyze_played_hand(self, obs, card_mask, chip_score):
        """
        Analyze played cards to determine hand type and scoring breakdown.

        Uses the actual chip score to reverse-engineer which hand was made,
        then shows the calculation breakdown.

        Args:
            obs: Observation with card_ranks and card_suits
            card_mask: Which cards were played
            chip_score: Actual chips scored (from C++)

        Returns:
            Dict with 'hand_type', 'hand_name', 'base_chips', 'base_mult',
            'rank_sum', 'contributing_cards', 'formula'
        """
        # Get selected card ranks
        selected_ranks = []
        selected_cards = []
        for i, (rank, suit, selected) in enumerate(zip(obs['card_ranks'], obs['card_suits'], card_mask)):
            if selected:
                selected_ranks.append(int(rank))
                selected_cards.append((int(rank), int(suit)))

        if not selected_ranks:
            return None

        # Try to determine hand type by checking each possible hand
        # and seeing which one matches the actual score
        for hand_type in range(8, -1, -1):  # Check from best to worst
            base_chips, base_mult = HAND_SCORING[hand_type]

            # Calculate what rank_sum would need to be for this hand type
            # score = (base_chips + rank_sum) * base_mult
            # rank_sum = (score / base_mult) - base_chips
            if base_mult == 0:
                continue

            if chip_score % base_mult != 0:
                continue

            implied_rank_sum = (chip_score // base_mult) - base_chips

            # Now figure out which cards contribute based on hand type
            contributing_cards, calculated_rank_sum = self._get_contributing_cards(
                selected_cards, hand_type
            )

            if calculated_rank_sum == implied_rank_sum:
                # Found the matching hand type!
                return {
                    'hand_type': hand_type,
                    'hand_name': HAND_TYPE_NAMES[hand_type],
                    'base_chips': base_chips,
                    'base_mult': base_mult,
                    'rank_sum': calculated_rank_sum,
                    'contributing_cards': contributing_cards,
                    'formula': f"({base_chips} + {calculated_rank_sum}) x {base_mult} = {chip_score}"
                }

        # Fallback: couldn't determine exact hand, show what we know
        return None

    def _get_contributing_cards(self, cards, hand_type):
        """
        Determine which cards contribute to scoring for a given hand type.

        Args:
            cards: List of (rank, suit) tuples
            hand_type: HandType enum value

        Returns:
            (contributing_cards, rank_sum) tuple
        """
        from collections import Counter

        ranks = [c[0] for c in cards]
        suits = [c[1] for c in cards]
        rank_counts = Counter(ranks)

        contributing = []
        rank_sum = 0

        if hand_type == 0:  # HIGH_CARD - only highest card
            if ranks:
                max_rank = max(ranks)
                contributing = [(max_rank, suits[ranks.index(max_rank)])]
                rank_sum = RANK_VALUES[max_rank]

        elif hand_type == 1:  # PAIR - only the 2 paired cards
            for rank, count in rank_counts.items():
                if count >= 2:
                    # Find the pair
                    pair_cards = [(r, s) for r, s in cards if r == rank][:2]
                    contributing = pair_cards
                    rank_sum = RANK_VALUES[rank] * 2
                    break

        elif hand_type == 2:  # TWO_PAIR - only the 4 cards in both pairs
            pairs = [r for r, c in rank_counts.items() if c >= 2]
            if len(pairs) >= 2:
                pairs = sorted(pairs, reverse=True)[:2]
                for pair_rank in pairs:
                    pair_cards = [(r, s) for r, s in cards if r == pair_rank][:2]
                    contributing.extend(pair_cards)
                    rank_sum += RANK_VALUES[pair_rank] * 2

        elif hand_type == 3:  # THREE_OF_A_KIND - only the 3 matching cards
            for rank, count in rank_counts.items():
                if count >= 3:
                    trips_cards = [(r, s) for r, s in cards if r == rank][:3]
                    contributing = trips_cards
                    rank_sum = RANK_VALUES[rank] * 3
                    break

        elif hand_type == 7:  # FOUR_OF_A_KIND - only the 4 matching cards
            for rank, count in rank_counts.items():
                if count >= 4:
                    quad_cards = [(r, s) for r, s in cards if r == rank][:4]
                    contributing = quad_cards
                    rank_sum = RANK_VALUES[rank] * 4
                    break

        else:  # STRAIGHT, FLUSH, FULL_HOUSE, STRAIGHT_FLUSH - all 5 cards
            contributing = cards[:5]
            rank_sum = sum(RANK_VALUES[r] for r, s in contributing)

        return contributing, rank_sum

    def format_scoring_breakdown(self, analysis):
        """
        Format the scoring analysis as a readable string.

        Args:
            analysis: Dict from analyze_played_hand()

        Returns:
            Multi-line string showing the scoring breakdown
        """
        if analysis is None:
            return "    (Unable to determine hand breakdown)"

        lines = []
        lines.append(f"    Hand: {analysis['hand_name']}")

        # Show contributing cards
        contrib_strs = [self.format_card(r, s) for r, s in analysis['contributing_cards']]
        lines.append(f"    Scoring cards: [{', '.join(contrib_strs)}]")

        # Show formula
        lines.append(f"    Calculation: {analysis['formula']}")

        return '\n'.join(lines)

    def visualize_step(self, obs, action, next_obs, reward, info, verbose=True, decision=None):
        """
        Display a single step with full decision context.

        Args:
            obs: Observation before action
            action: Action dict with 'type' and 'card_mask'
            next_obs: Observation after action
            reward: Reward received
            info: Info dict with 'raw_reward', 'win', etc.
            verbose: If True, show detailed information
            decision: Optional agent decision string (e.g., "PLAY -> PLAY_BEST_HAND")
        """
        self.step_count += 1

        # Extract state information
        plays_left = obs['plays_left']
        discards_left = obs['discards_left']
        chips_before = obs['chips'][0]
        chips_after = next_obs['chips'][0]
        target = chips_before + obs['chips_to_target'][0]  # Reconstruct target

        # Format action
        action_type = "PLAY" if action['type'] == 0 else "DISCARD"
        selected_cards = self.format_selected_cards(obs, action['card_mask'])
        num_selected = sum(action['card_mask'])

        # Display step header
        print(f"\n{'='*70}")
        print(f"STEP {self.step_count}")
        print(f"{'='*70}")

        # State before action
        print(f"Resources: {plays_left} plays, {discards_left} discards remaining")
        if decision:
            print(f"Decision: {decision}")
        print(f"Progress:  {chips_before}/{target} chips ({chips_before/target*100:.1f}%)")
        print(f"\nHand dealt: {self.format_hand(obs)}")

        if verbose and obs.get('best_hand_type') is not None:
            best_hand = self.get_hand_type_name(obs['best_hand_type'])
            best_score = obs['best_hand_score'][0]
            print(f"Best possible: {best_hand} ({best_score} chips)")

        # Action taken
        print(f"\n>>> ACTION: {action_type} {num_selected} card(s)")
        print(f"    Selected: {selected_cards}")

        # Result
        raw_chip_gain = int(info.get('raw_reward', 0))
        arrow = "->" if self.use_ascii else "→"

        if action['type'] == 0:  # PLAY action
            print(f"\n<<< RESULT:")
            if raw_chip_gain > 0:
                # Analyze the hand and show scoring breakdown
                analysis = self.analyze_played_hand(obs, action['card_mask'], raw_chip_gain)
                if analysis:
                    print(self.format_scoring_breakdown(analysis))
                else:
                    print(f"    Chips scored: +{raw_chip_gain}")
                print(f"    Total: {chips_before} {arrow} {chips_after}")
            else:
                print(f"    No chips scored (invalid play or 0 points)")
        else:  # DISCARD action
            print(f"\n<<< RESULT:")
            print(f"    Discarded {num_selected} card(s)")
            # Note: After discard, new cards are dealt, but we don't show them here
            # (they'll appear in next_obs for the next step)

        # Reward information
        print(f"\nReward: {reward:.1f}", end="")
        if abs(reward - raw_chip_gain) > 0.1:  # If shaped reward differs from raw
            print(f" (raw: {raw_chip_gain:.1f})")
        else:
            print()

        # Episode end
        if info.get('win', False):
            print(f"\n{'*'*70}")
            if self.use_ascii:
                print(f"*** WIN! Target reached: {chips_after}/{target}")
            else:
                print(f"🎉 WIN! Target reached: {chips_after}/{target}")
            print(f"{'*'*70}")
        elif next_obs['plays_left'] == 0 and next_obs['discards_left'] == 0:
            print(f"\n{'*'*70}")
            if self.use_ascii:
                print(f"XXX LOSS. Final score: {chips_after}/{target}")
            else:
                print(f"💀 LOSS. Final score: {chips_after}/{target}")
            print(f"{'*'*70}")

    def visualize_step_compact(self, obs, action, next_obs, reward, info, decision=None):
        """
        Display a compact one-line summary of the step.

        Args:
            obs: Observation before action
            action: Action dict
            next_obs: Observation after action
            reward: Reward received
            info: Info dict
            decision: Optional agent decision string
        """
        self.step_count += 1

        action_type = "PLAY" if action['type'] == 0 else "DISC"
        num_selected = sum(action['card_mask'])
        selected = self.format_selected_cards(obs, action['card_mask'])

        chips_before = obs['chips'][0]
        chips_after = next_obs['chips'][0]
        chip_gain = info.get('raw_reward', 0)

        status = ""
        if info.get('win', False):
            status = " [WIN!]"
        elif next_obs['plays_left'] == 0 and next_obs['discards_left'] == 0:
            status = " [LOSS]"

        arrow = "->" if self.use_ascii else "→"
        decision_str = f" [{decision}]" if decision else ""
        print(f"Step {self.step_count:2d}: {action_type} {selected:30s} {arrow} "
              f"+{chip_gain:4d} chips (total: {chips_after:4d}) | "
              f"R={reward:6.1f}{status}{decision_str}")

    def print_episode_header(self, episode_num, target_score, seed=None):
        """
        Print header for episode visualization.

        Args:
            episode_num: Episode number
            target_score: Target score for the episode
            seed: Random seed (optional)
        """
        print(f"\n{'#'*70}")
        print(f"# EPISODE {episode_num}")
        if seed is not None:
            print(f"# Seed: {seed}")
        print(f"# Target Score: {target_score}")
        print(f"{'#'*70}")

    def print_episode_summary(self, result):
        """
        Print summary statistics for completed episode.

        Args:
            result: Dict with 'win', 'steps', 'total_reward', 'final_chips', etc.
        """
        print(f"\n{'='*70}")
        print(f"EPISODE SUMMARY")
        print(f"{'='*70}")
        print(f"Result:       {'WIN' if result.get('win', False) else 'LOSS'}")
        print(f"Steps taken:  {result.get('steps', 0)}")
        print(f"Final chips:  {result.get('final_chips', 0)}")
        print(f"Total reward: {result.get('total_reward', 0):.1f}")
        print(f"{'='*70}\n")


def test_visualizer():
    """Test the visualizer with a sample observation."""
    from balatro_env import BalatroBatchedSimEnv
    import numpy as np

    # Create environment
    env = BalatroBatchedSimEnv(target_score=300)
    obs, _ = env.reset(seed=42)

    # Create visualizer
    viz = AgentVisualizer()
    viz.reset_episode()
    viz.print_episode_header(1, 300, seed=42)

    # Generate a random action
    action = {
        'type': 0,  # PLAY
        'card_mask': np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=np.int8)
    }

    # Execute action
    next_obs, reward, terminated, truncated, info = env.step(action)

    # Visualize
    viz.visualize_step(obs, action, next_obs, reward, info)

    print("\n" + "="*70)
    print("Compact mode:")
    print("="*70)

    # Reset and show compact version
    obs, _ = env.reset(seed=42)
    viz.reset_episode()
    next_obs, reward, terminated, truncated, info = env.step(action)
    viz.visualize_step_compact(obs, action, next_obs, reward, info)


if __name__ == "__main__":
    test_visualizer()
