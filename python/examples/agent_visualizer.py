#!/usr/bin/env python3
"""
Agent decision visualization utilities.

Provides tools to display agent decisions including:
- Hand dealt (cards with ranks and suits)
- Action chosen (play/discard with selected cards)
- Result (points scored, hand type)
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

    def visualize_step(self, obs, action, next_obs, reward, info, verbose=True):
        """
        Display a single step with full decision context.

        Args:
            obs: Observation before action
            action: Action dict with 'type' and 'card_mask'
            next_obs: Observation after action
            reward: Reward received
            info: Info dict with 'raw_reward', 'win', etc.
            verbose: If True, show detailed information
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
        raw_chip_gain = info.get('raw_reward', 0)

        if action['type'] == 0:  # PLAY action
            # Try to determine what hand was played (if we have the info)
            # For now, we'll just show the chip gain
            arrow = "->" if self.use_ascii else "→"
            print(f"\n<<< RESULT:")
            if raw_chip_gain > 0:
                print(f"    Chips scored: +{raw_chip_gain}")
                print(f"    Total chips: {chips_before} {arrow} {chips_after}")
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

    def visualize_step_compact(self, obs, action, next_obs, reward, info):
        """
        Display a compact one-line summary of the step.

        Args:
            obs: Observation before action
            action: Action dict
            next_obs: Observation after action
            reward: Reward received
            info: Info dict
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
        print(f"Step {self.step_count:2d}: {action_type} {selected:30s} {arrow} "
              f"+{chip_gain:4d} chips (total: {chips_after:4d}) | "
              f"R={reward:6.1f}{status}")

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
