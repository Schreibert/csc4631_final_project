#!/usr/bin/env python3
"""
Strategy-Based Action Encoder for Balatro Poker Q-Learning.

This module provides the StrategyActionEncoder class that converts high-level
poker strategies into concrete card selection actions. Instead of choosing from
512 possible card combinations (2^8 - illegal states), the agent selects from
just 5 meaningful strategies, dramatically reducing Q-table size.

Action Space (5 strategies):
    0: PLAY_BEST_HAND        - Play optimal 5-card hand (always available if plays > 0)
    1: DISCARD_UPGRADE       - Keep best-hand cards, discard others to try for upgrade
    2: DISCARD_FLUSH_CHASE   - Chase flush by discarding non-flush-suit cards
    3: DISCARD_STRAIGHT_CHASE - Chase straight by discarding non-consecutive cards
    4: DISCARD_AGGRESSIVE    - Discard 5 lowest cards for complete reset

Why Strategy-Based Actions:
    - Reduces action space from ~512 to 5 (99% reduction)
    - Encodes domain knowledge (poker strategy) into action design
    - Each action has clear semantic meaning
    - Enables effective Q-learning with reasonable exploration

Algorithm Details:
    - DISCARD_FLUSH_CHASE: Identifies suit with 4+ cards, discards all others
    - DISCARD_STRAIGHT_CHASE: Finds longest consecutive rank sequence, including
      A-2-3-4-5 "wheel" detection where Ace acts as low card
    - DISCARD_UPGRADE: Uses C++ enumerate_all_actions() to find best hand indices

Usage:
    Command line (prints action summary):
        python strategy_action_encoder.py

    Programmatic:
        >>> from balatro_env import BalatroBatchedSimEnv
        >>> env = BalatroBatchedSimEnv(target_score=300)
        >>> encoder = StrategyActionEncoder(env)
        >>>
        >>> # Get valid actions for current state
        >>> obs, _ = env.reset(seed=42)
        >>> valid = encoder.get_valid_actions(obs)
        >>> print(f"Valid actions: {valid}")  # e.g., [0, 1, 4]
        >>>
        >>> # Convert strategy to environment action
        >>> action = encoder.decode_to_env_action(0, obs)  # PLAY_BEST_HAND
        >>> print(f"Action type: {'PLAY' if action['type'] == 0 else 'DISCARD'}")
        >>> print(f"Card mask: {action['card_mask']}")

Contributors:
    Tyler Schreiber, Alec Nartatez
"""

import numpy as np
from typing import List, Set, Dict, Any


class StrategyActionEncoder:
    """
    Encodes strategy-based actions for Q-table indexing.

    This class bridges the gap between high-level strategy decisions and
    concrete card selection actions that the environment expects.

    Action Space (5 strategies):
        0: PLAY_BEST_HAND - Play optimal 5-card hand (uses C++ evaluation)
        1: DISCARD_UPGRADE - Keep best-hand cards, discard others to upgrade
        2: DISCARD_FLUSH_CHASE - Discard non-flush-suit cards (requires 4+ same suit)
        3: DISCARD_STRAIGHT_CHASE - Discard non-consecutive cards (handles wheel)
        4: DISCARD_AGGRESSIVE - Discard 5 lowest cards, complete hand reset

    Action Validity Rules:
        - PLAY_BEST_HAND: Valid when plays_left > 0
        - DISCARD_*: Valid when discards_left > 0
        - FLUSH_CHASE: Only valid when flush_potential=1 and best_hand_type < FLUSH
        - STRAIGHT_CHASE: Only valid when straight_potential=1 and best_hand_type < STRAIGHT

    Example:
        >>> encoder = StrategyActionEncoder(env)
        >>> valid = encoder.get_valid_actions(obs)
        >>> action = encoder.decode_to_env_action(valid[0], obs)
    """

    NUM_ACTIONS = 5

    # Action constants
    PLAY_BEST_HAND = 0
    DISCARD_UPGRADE = 1
    DISCARD_FLUSH_CHASE = 2
    DISCARD_STRAIGHT_CHASE = 3
    DISCARD_AGGRESSIVE = 4

    ACTION_NAMES = [
        "PLAY_BEST_HAND",
        "DISCARD_UPGRADE",
        "DISCARD_FLUSH_CHASE",
        "DISCARD_STRAIGHT_CHASE",
        "DISCARD_AGGRESSIVE"
    ]

    ACTION_DESCRIPTIONS = [
        "Play optimal 5-card hand",
        "Keep best-hand cards, discard others",
        "Chase flush (discard non-flush cards)",
        "Chase straight (discard non-consecutive)",
        "Aggressive reset (discard 5 lowest)"
    ]

    def __init__(self, env):
        """
        Initialize with environment reference for best hand lookup.

        Args:
            env: BalatroBatchedSimEnv instance for get_best_hand()
        """
        self.env = env

    def get_num_actions(self) -> int:
        """Return total number of strategy actions."""
        return self.NUM_ACTIONS

    def get_action_name(self, action_idx: int) -> str:
        """Get human-readable action name."""
        if 0 <= action_idx < self.NUM_ACTIONS:
            return self.ACTION_NAMES[action_idx]
        return f"UNKNOWN({action_idx})"

    def get_action_description(self, action_idx: int) -> str:
        """Get action description."""
        if 0 <= action_idx < self.NUM_ACTIONS:
            return self.ACTION_DESCRIPTIONS[action_idx]
        return "Unknown action"

    def get_valid_actions(self, obs: Dict[str, Any]) -> List[int]:
        """
        Get list of valid action indices for current observation.

        Args:
            obs: Current observation dict with plays_left, discards_left,
                 flush_potential, straight_potential, best_hand_type

        Returns:
            List of valid action indices (0-4)
        """
        valid = []
        plays_left = obs['plays_left']
        discards_left = obs['discards_left']

        # Action 0: PLAY_BEST_HAND - valid if plays available
        if plays_left > 0:
            valid.append(self.PLAY_BEST_HAND)

        # Action 1: DISCARD_UPGRADE - valid if discards available
        if discards_left > 0:
            valid.append(self.DISCARD_UPGRADE)

        # Action 2: DISCARD_FLUSH_CHASE - valid if flush potential and not already flush+
        if discards_left > 0 and obs.get('flush_potential', 0) == 1:
            if obs.get('best_hand_type', 0) < 5:  # < FLUSH
                valid.append(self.DISCARD_FLUSH_CHASE)

        # Action 3: DISCARD_STRAIGHT_CHASE - valid if straight potential and not already straight+
        if discards_left > 0 and obs.get('straight_potential', 0) == 1:
            if obs.get('best_hand_type', 0) < 4:  # < STRAIGHT
                valid.append(self.DISCARD_STRAIGHT_CHASE)

        # Action 4: DISCARD_AGGRESSIVE - always valid if discards available
        if discards_left > 0:
            valid.append(self.DISCARD_AGGRESSIVE)

        # Fallback: ensure at least one action is valid
        if not valid:
            if plays_left > 0:
                valid.append(self.PLAY_BEST_HAND)
            elif discards_left > 0:
                valid.append(self.DISCARD_UPGRADE)

        return valid

    def decode_to_env_action(self, action_idx: int, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert strategy action index to environment action dict.

        Args:
            action_idx: Strategy action (0-4)
            obs: Current observation with card_ranks, card_suits, etc.

        Returns:
            Action dict with 'type' (0=PLAY, 1=DISCARD) and 'card_mask' (8 bools)
        """
        if action_idx == self.PLAY_BEST_HAND:
            return self._play_best_hand(obs)
        elif action_idx == self.DISCARD_UPGRADE:
            return self._discard_upgrade(obs)
        elif action_idx == self.DISCARD_FLUSH_CHASE:
            return self._discard_flush_chase(obs)
        elif action_idx == self.DISCARD_STRAIGHT_CHASE:
            return self._discard_straight_chase(obs)
        elif action_idx == self.DISCARD_AGGRESSIVE:
            return self._discard_aggressive(obs)
        else:
            raise ValueError(f"Unknown action index: {action_idx}")

    def _get_best_play_action(self) -> Dict[str, Any]:
        """
        Get the best PLAY action using C++ enumerate_all_actions.

        Returns:
            Dict with 'type' and 'card_mask' for best play
        """
        # Use C++ to enumerate all actions and find best play
        all_actions = self.env.sim.enumerate_all_actions()

        best_play = None
        best_chips = -1

        for outcome in all_actions:
            if not outcome.valid:
                continue
            # Check if it's a PLAY action (type 0)
            if outcome.action.type.value == 0:  # PLAY
                if outcome.predicted_chips > best_chips:
                    best_chips = outcome.predicted_chips
                    best_play = outcome.action

        if best_play is None:
            # Fallback: play first 5 cards
            card_mask = np.zeros(8, dtype=np.int8)
            card_mask[:5] = 1
            return {'type': 0, 'card_mask': card_mask}

        # Convert C++ action to dict
        card_mask = np.array([1 if m else 0 for m in best_play.card_mask], dtype=np.int8)
        return {'type': 0, 'card_mask': card_mask}

    def _get_best_hand_indices(self, obs: Dict[str, Any]) -> Set[int]:
        """
        Get hand indices (0-7) of cards forming the best hand.

        Uses enumerate_all_actions to find best play and extract card mask.

        Args:
            obs: Observation with card_ranks and card_suits arrays

        Returns:
            Set of indices (0-7) for cards in best hand
        """
        best_action = self._get_best_play_action()
        return set(i for i, m in enumerate(best_action['card_mask']) if m)

    def _play_best_hand(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Play the best 5-card hand.

        Uses C++ enumerate_all_actions() to find optimal cards.
        """
        return self._get_best_play_action()

    def _discard_upgrade(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Discard all cards NOT in best hand to try for upgrade.

        Example: With two pair (KK QQ), keeps those 4 cards and discards 4 others.
        """
        best_hand_indices = self._get_best_hand_indices(obs)

        # All non-best-hand cards are discard candidates
        discard_indices = [i for i in range(8) if i not in best_hand_indices]

        # Cap at 5 discards (game limit)
        num_to_discard = min(5, len(discard_indices))

        # If we can discard more than available non-best cards, that's fine
        # Just discard what we can
        card_mask = np.zeros(8, dtype=np.int8)
        for idx in discard_indices[:num_to_discard]:
            card_mask[idx] = 1

        # Must discard at least 1 card
        if num_to_discard == 0:
            # Edge case: all cards are in best hand (shouldn't happen with 8 cards, 5 in hand)
            # Discard lowest ranked card
            ranks = list(obs['card_ranks'])
            lowest_idx = min(range(8), key=lambda i: ranks[i])
            card_mask[lowest_idx] = 1

        return {'type': 1, 'card_mask': card_mask}

    def _discard_flush_chase(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Discard non-flush-suit cards to chase a flush.

        Finds the suit with 4+ cards and discards everything else.
        """
        from collections import Counter

        suits = list(obs['card_suits'])
        ranks = list(obs['card_ranks'])
        suit_counts = Counter(suits)

        # Find the flush suit (4+ cards)
        flush_suit = None
        for suit, count in suit_counts.items():
            if count >= 4:
                flush_suit = suit
                break

        if flush_suit is None:
            # No flush potential - fall back to upgrade
            return self._discard_upgrade(obs)

        # Identify cards NOT of flush suit
        non_flush_cards = [(i, ranks[i]) for i in range(8) if suits[i] != flush_suit]

        # Sort by rank (discard lowest first if we need to cap)
        non_flush_cards.sort(key=lambda x: x[1])

        # Discard up to 5 non-flush cards
        num_to_discard = min(5, len(non_flush_cards))

        card_mask = np.zeros(8, dtype=np.int8)
        for idx, _ in non_flush_cards[:num_to_discard]:
            card_mask[idx] = 1

        # Must discard at least 1
        if num_to_discard == 0:
            # All cards are flush suit - discard lowest ranked one
            lowest_idx = min(range(8), key=lambda i: ranks[i])
            card_mask[lowest_idx] = 1

        return {'type': 1, 'card_mask': card_mask}

    def _discard_straight_chase(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Discard non-consecutive cards to chase a straight.

        Finds the longest consecutive rank sequence and discards others.

        Algorithm:
            1. Group card indices by rank
            2. Find longest consecutive sequence using linear scan
            3. Check for A-2-3-4-5 "wheel" potential:
               - Ace is rank 12, 2-5 are ranks 0-3
               - If Ace present and 3+ low cards (2,3,4,5), treat as wheel draw
            4. Keep one card per rank in best sequence
            5. Discard all other cards (lowest ranks first, cap at 5)

        Wheel Detection:
            The A-2-3-4-5 "wheel" straight requires special handling because
            the Ace (rank 12) wraps around to act as rank 0. We detect this
            by checking if we have an Ace plus at least 3 cards from ranks 0-3
            (representing 2, 3, 4, 5).

        Returns:
            Dict with type=1 (DISCARD) and card_mask marking cards to discard.
        """
        ranks = list(obs['card_ranks'])

        # Find indices grouped by rank
        rank_to_indices = {}
        for i, r in enumerate(ranks):
            if r not in rank_to_indices:
                rank_to_indices[r] = []
            rank_to_indices[r].append(i)

        # Get unique sorted ranks
        unique_ranks = sorted(set(ranks))

        # Find longest consecutive sequence
        best_seq = []
        current_seq = [unique_ranks[0]] if unique_ranks else []

        for i in range(1, len(unique_ranks)):
            if unique_ranks[i] == unique_ranks[i - 1] + 1:
                current_seq.append(unique_ranks[i])
            else:
                if len(current_seq) > len(best_seq):
                    best_seq = current_seq
                current_seq = [unique_ranks[i]]

        if len(current_seq) > len(best_seq):
            best_seq = current_seq

        # Check for A-2-3-4 wheel potential (ace as low)
        # Ace is rank 12, 2 is rank 0, etc.
        if 12 in unique_ranks:  # Have an ace
            wheel_ranks = [r for r in unique_ranks if r <= 3 or r == 12]
            # Check if we have A + some of 2,3,4,5 (ranks 0,1,2,3)
            low_ranks = [r for r in wheel_ranks if r <= 3]
            if len(low_ranks) >= 3:  # At least 3 low cards + ace
                wheel_seq = sorted(low_ranks) + [12]
                if len(wheel_seq) > len(best_seq):
                    best_seq = wheel_seq

        # Keep indices that are part of the straight draw (one per rank in sequence)
        keep_indices = set()
        for r in best_seq:
            if r in rank_to_indices:
                # Keep just one card per rank
                keep_indices.add(rank_to_indices[r][0])

        # Discard everything else
        discard_indices = [i for i in range(8) if i not in keep_indices]

        # Sort by rank and cap at 5
        discard_with_ranks = [(i, ranks[i]) for i in discard_indices]
        discard_with_ranks.sort(key=lambda x: x[1])

        num_to_discard = min(5, len(discard_with_ranks))

        card_mask = np.zeros(8, dtype=np.int8)
        for idx, _ in discard_with_ranks[:num_to_discard]:
            card_mask[idx] = 1

        # Must discard at least 1
        if num_to_discard == 0:
            lowest_idx = min(range(8), key=lambda i: ranks[i])
            card_mask[lowest_idx] = 1

        return {'type': 1, 'card_mask': card_mask}

    def _discard_aggressive(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggressive discard: remove 5 lowest-ranked cards regardless of hand.

        Used when hand is weak (high card, low pair) for a complete reset.
        """
        ranks = list(obs['card_ranks'])

        # Get all cards with indices, sorted by rank
        cards_with_indices = [(i, ranks[i]) for i in range(8)]
        cards_with_indices.sort(key=lambda x: x[1])

        # Discard the 5 lowest
        discard_indices = [idx for idx, _ in cards_with_indices[:5]]

        card_mask = np.zeros(8, dtype=np.int8)
        for idx in discard_indices:
            card_mask[idx] = 1

        return {'type': 1, 'card_mask': card_mask}


# Utility functions for testing/debugging

def print_action_summary():
    """Print summary of available strategy actions."""
    print("=" * 60)
    print("STRATEGY-BASED ACTION SPACE")
    print("=" * 60)
    print(f"{'ID':<4} {'Name':<25} {'Description':<30}")
    print("-" * 60)
    for i in range(StrategyActionEncoder.NUM_ACTIONS):
        print(f"{i:<4} {StrategyActionEncoder.ACTION_NAMES[i]:<25} "
              f"{StrategyActionEncoder.ACTION_DESCRIPTIONS[i]:<30}")
    print("=" * 60)


if __name__ == "__main__":
    print_action_summary()
