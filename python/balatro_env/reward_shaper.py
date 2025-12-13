"""
Reward shaping for Balatro Poker RL environment.

This module provides the RewardShaper class which loads reward configurations
from YAML and computes shaped rewards based on game state and actions.

Features:
    - YAML-based configuration for easy experimentation
    - Terminal rewards (win bonus, loss penalty)
    - Efficiency rewards (play conservation, step penalty)
    - Progress rewards (chip gain, threshold bonuses)
    - Hand quality bonuses (small rewards for strong hands)
    - Discard shaping (rewards for strategic discarding)

Usage:
    >>> # Default config (loads rewards_config.yaml)
    >>> shaper = RewardShaper()
    >>>
    >>> # Custom config file
    >>> shaper = RewardShaper(config_path='my_rewards.yaml')
    >>>
    >>> # Programmatic config
    >>> config = {'outcome': {'win_bonus': 100}, ...}
    >>> shaper = RewardShaper(config=config)
    >>>
    >>> # Use in episode loop
    >>> shaper.reset_episode_state()
    >>> reward = shaper.shape_reward(
    ...     raw_chip_delta=50,
    ...     current_chips=150,
    ...     target_score=300,
    ...     action=0,  # PLAY_BEST_HAND
    ...     plays_left=3,
    ...     discards_left=2,
    ...     done=False,
    ...     win=False
    ... )

See rewards_config.yaml for the full configuration structure.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List


class RewardShaper:
    """
    Computes shaped rewards for the Balatro poker environment.

    This class handles:
    - Loading reward configuration from YAML
    - Tracking episode state for threshold bonuses
    - Computing shaped rewards from raw chip deltas
    - Terminal bonuses/penalties for wins/losses
    """

    # Map action codes to hand types (for hand quality bonuses)
    # Based on action space definition in CLAUDE.md
    ACTION_TO_HAND_TYPE = {
        0: None,  # "Best scoring hand" - we'll infer from chips gained
        1: "pair",  # "Play highest-value Pair"
        2: None,  # "Discard non-paired cards"
        3: None,  # "Discard 3 lowest cards"
        4: "high_card",  # "Play 3 highest as High Card"
        5: None,  # "Random valid play"
        6: None,  # "Random valid discard"
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        """
        Initialize RewardShaper with configuration.

        Args:
            config: Dict with reward configuration (overrides config_path)
            config_path: Path to YAML config file (default: rewards_config.yaml in same dir)
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(Path(config_path))
        else:
            # Default: load from same directory as this file
            default_path = Path(__file__).parent / "rewards_config.yaml"
            self.config = self._load_config(default_path)

        # Validate config structure
        self._validate_config()

        # Episode tracking state (reset per episode)
        self.reset_episode_state()

    def _load_config(self, path: Path) -> Dict[str, Any]:
        """Load reward configuration from YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Reward config file not found: {path}")

        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def _validate_config(self):
        """Validate that required config keys exist."""
        required_keys = ['outcome', 'efficiency', 'progress', 'hand_quality', 'penalties']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def reset_episode_state(self):
        """Reset per-episode tracking state (call at start of each episode)."""
        # Track which thresholds have already been crossed
        self.thresholds_crossed = set()
        # Track previous chip count for threshold detection
        self.previous_chips = 0
        # Track previous best hand type for hand improvement rewards
        self.previous_best_hand_type = 0

    def shape_reward(
        self,
        raw_chip_delta: float,
        current_chips: int,
        target_score: int,
        action: int,
        plays_left: int,
        discards_left: int,
        done: bool,
        win: bool,
        hand_type: Optional[str] = None,
        # New parameters for discard shaping
        best_hand_type: int = 0,
        flush_potential: bool = False,
        straight_potential: bool = False,
        is_discard_action: bool = False,
        discards_left_before: int = 0
    ) -> float:
        """
        Compute shaped reward from game state and raw chip delta.

        Args:
            raw_chip_delta: Raw chip gain from C++ simulator
            current_chips: Current total chips accumulated
            target_score: Target score to beat
            action: Action code taken (0-6)
            plays_left: Remaining plays
            discards_left: Remaining discards (after action)
            done: Whether episode is finished
            win: Whether episode was won
            hand_type: Optional hand type string (e.g., "pair", "flush")
            best_hand_type: Current best hand type (0=HC, 1=Pair, ..., 8=SF)
            flush_potential: Whether hand has flush potential
            straight_potential: Whether hand has straight potential
            is_discard_action: Whether action was a discard
            discards_left_before: Discards remaining before action (for penalty calc)

        Returns:
            Shaped reward value
        """
        reward = 0.0

        # 1. Base chip progress reward (scaled and normalized)
        if raw_chip_delta > 0:
            chip_reward = (raw_chip_delta * self.config['progress']['chip_gain_scale']
                          / self.config['progress']['chip_normalization'])
            reward += chip_reward

        # 2. Step penalty (small cost per action)
        reward -= self.config['efficiency']['step_penalty']

        # 3. Threshold crossing bonuses (one-time per threshold)
        if not done:  # Only give threshold bonuses during episode
            threshold_bonus = self._compute_threshold_bonus(
                self.previous_chips, current_chips, target_score
            )
            reward += threshold_bonus

        # 4. Hand quality bonus (encourage playing strong hands)
        if self.config['hand_quality']['enabled'] and raw_chip_delta > 0:
            hand_bonus = self._compute_hand_quality_bonus(action, hand_type, raw_chip_delta)
            reward += hand_bonus

        # 5. Desperate play penalty (playing weak hands when low on plays)
        if not done and plays_left < 2 and action == 4:  # Action 4 = play high card
            progress_fraction = current_chips / target_score
            if progress_fraction < self.config['penalties']['desperate_threshold']:
                reward -= self.config['penalties']['desperate_play']

        # 6. Discard shaping rewards/penalties
        discard_cfg = self.config.get('discard_shaping', {})
        if discard_cfg.get('enabled', False) and not done:
            # 6a. Smart discard bonus (reward discarding when hand is weak)
            if is_discard_action:
                reward += self._compute_discard_bonus(
                    self.previous_best_hand_type,
                    flush_potential,
                    straight_potential,
                    discard_cfg
                )
                # 6b. Hand improvement reward (reward discards that improve hand)
                reward += self._compute_hand_improvement_reward(
                    self.previous_best_hand_type,
                    best_hand_type,
                    discard_cfg
                )
                # 6c. Valuable hand discard penalty (penalize discarding strong hands)
                valuable_threshold = discard_cfg.get('valuable_hand_threshold', 4)
                if self.previous_best_hand_type >= valuable_threshold:
                    reward -= discard_cfg.get('valuable_hand_discard_penalty', 0.0)
            else:
                # 6d. Weak hand play penalty (penalize playing weak hands with discards available)
                reward -= self._compute_weak_play_penalty(
                    self.previous_best_hand_type,
                    discards_left_before,
                    discard_cfg
                )
                # 6e. Doomed play penalty (using last play when discards left and can't win)
                if plays_left == 0 and discards_left_before > 0:
                    if current_chips < target_score:
                        reward -= discard_cfg.get('doomed_play_penalty', 0.0)

        # 7. Terminal rewards (win/loss bonuses)
        if done:
            if win:
                # Win bonus + play conservation bonus
                reward += self.config['outcome']['win_bonus']
                conservation_bonus = plays_left * self.config['efficiency']['play_conservation_bonus']
                reward += conservation_bonus

                # Optional: safety margin bonus
                if self.config['advanced']['safety_margin_bonus']['enabled']:
                    margin = current_chips - target_score
                    if margin > 0:
                        margin_bonus = min(
                            margin * self.config['advanced']['safety_margin_bonus']['per_chip_over_target'],
                            self.config['advanced']['safety_margin_bonus']['max_bonus']
                        )
                        reward += margin_bonus
            else:
                # Loss penalty
                reward -= self.config['outcome']['loss_penalty']
                # Unused discards penalty (penalize wasted discards at loss)
                discard_cfg = self.config.get('discard_shaping', {})
                if discard_cfg.get('enabled', False) and discards_left > 0:
                    unused_penalty = discards_left * discard_cfg.get('unused_discards_penalty', 0.0)
                    reward -= unused_penalty

        # Update tracking state
        self.previous_chips = current_chips
        self.previous_best_hand_type = best_hand_type

        return reward

    def _compute_threshold_bonus(
        self,
        prev_chips: int,
        current_chips: int,
        target_score: int
    ) -> float:
        """
        Compute bonus for crossing target score thresholds.

        Only awards each threshold bonus once per episode.
        """
        bonus = 0.0

        for threshold_config in self.config['progress']['target_threshold_bonuses']:
            threshold_frac = threshold_config['threshold']
            threshold_value = target_score * threshold_frac

            # Check if we crossed this threshold on this step
            if prev_chips < threshold_value <= current_chips:
                # Only award if not already crossed
                if threshold_frac not in self.thresholds_crossed:
                    bonus += threshold_config['bonus']
                    self.thresholds_crossed.add(threshold_frac)

        return bonus

    def _compute_hand_quality_bonus(
        self,
        action: int,
        hand_type: Optional[str],
        raw_chip_delta: float
    ) -> float:
        """
        Compute bonus for playing strong poker hands.

        Uses hand_type if provided, otherwise infers from action.
        Chip-based inference only used for action 0 (best scoring hand).
        """
        bonuses = self.config['hand_quality']['bonuses']

        # If hand type explicitly provided, use it
        if hand_type is not None and hand_type != "":
            if hand_type in bonuses:
                return bonuses[hand_type]

        # Try to infer from action code
        inferred_type = self.ACTION_TO_HAND_TYPE.get(action)
        if inferred_type is not None and inferred_type in bonuses:
            return bonuses[inferred_type]

        # Fallback: For action 0 (play best hand), roughly infer from chip gain
        # This is approximate and only applies to generic "play best" actions
        if action == 0 and raw_chip_delta > 0:
            # Based on scoring formula in CLAUDE.md
            if raw_chip_delta >= 300:  # High value hands
                return bonuses.get('flush', 0)
            elif raw_chip_delta >= 200:
                return bonuses.get('three_of_a_kind', 0)
            elif raw_chip_delta >= 100:
                return bonuses.get('two_pair', 0)
            elif raw_chip_delta >= 50:
                return bonuses.get('pair', 0)
            else:
                return bonuses.get('high_card', 0)

        # No hand type bonus if we can't determine hand type
        return 0.0

    def _compute_discard_bonus(
        self,
        hand_type_before: int,
        flush_potential: bool,
        straight_potential: bool,
        discard_cfg: Dict[str, Any]
    ) -> float:
        """
        Compute bonus for smart discarding.

        Args:
            hand_type_before: Hand type before discard (0=HC, 1=Pair, etc.)
            flush_potential: Whether hand has flush potential
            straight_potential: Whether hand has straight potential
            discard_cfg: Discard shaping config section

        Returns:
            Bonus reward for discarding
        """
        bonus = 0.0
        weak_threshold = discard_cfg.get('weak_hand_threshold', 2)

        # Smart discard bonus: reward discarding weak hands
        if hand_type_before < weak_threshold:
            bonus += discard_cfg.get('smart_discard_bonus', 0.0)

        # Draw chase bonus: extra reward when chasing flush/straight
        if flush_potential or straight_potential:
            bonus += discard_cfg.get('draw_chase_bonus', 0.0)

        return bonus

    def _compute_hand_improvement_reward(
        self,
        hand_type_before: int,
        hand_type_after: int,
        discard_cfg: Dict[str, Any]
    ) -> float:
        """
        Compute reward for hand improvement after discard.

        Formula: reward = min(n * scale, max_reward)
        where n = hand_type_after - hand_type_before

        Args:
            hand_type_before: Hand type before discard
            hand_type_after: Hand type after discard
            discard_cfg: Discard shaping config section

        Returns:
            Improvement reward (0 if hand didn't improve)
        """
        improvement = hand_type_after - hand_type_before

        if improvement <= 0:
            return 0.0

        scale = discard_cfg.get('hand_improvement_scale', 0.1)
        max_reward = discard_cfg.get('hand_improvement_max', 0.5)

        return min(improvement * scale, max_reward)

    def _compute_weak_play_penalty(
        self,
        hand_type: int,
        discards_left: int,
        discard_cfg: Dict[str, Any]
    ) -> float:
        """
        Compute penalty for playing weak hands when discards are available.

        Args:
            hand_type: Current hand type (0=HC, 1=Pair, etc.)
            discards_left: Number of discards remaining
            discard_cfg: Discard shaping config section

        Returns:
            Penalty amount (0 if not applicable)
        """
        if discards_left <= 0:
            return 0.0

        weak_threshold = discard_cfg.get('weak_hand_threshold', 2)

        if hand_type < weak_threshold:
            return discard_cfg.get('weak_hand_play_penalty', 0.0)

        return 0.0

    def get_config_summary(self) -> str:
        """Return a human-readable summary of the reward configuration."""
        lines = [
            f"Reward Configuration (version: {self.config.get('version', 'unknown')})",
            f"Description: {self.config.get('description', 'No description')}",
            "",
            "Key Parameters:",
            f"  Win Bonus: {self.config['outcome']['win_bonus']}",
            f"  Loss Penalty: {self.config['outcome']['loss_penalty']}",
            f"  Play Conservation Bonus: {self.config['efficiency']['play_conservation_bonus']} per play",
            f"  Step Penalty: {self.config['efficiency']['step_penalty']}",
            f"  Hand Quality Bonuses: {'Enabled' if self.config['hand_quality']['enabled'] else 'Disabled'}",
        ]
        return "\n".join(lines)


# Convenience function for backwards compatibility
def create_legacy_reward_shaper(win_bonus: int, loss_penalty: int, step_penalty: float) -> RewardShaper:
    """
    Create a RewardShaper with legacy reward structure (pre-YAML config).

    This allows backwards compatibility with old code that passed rewards as constructor args.
    """
    legacy_config = {
        'outcome': {
            'win_bonus': win_bonus,
            'loss_penalty': loss_penalty,
        },
        'efficiency': {
            'play_conservation_bonus': 0,  # Not in legacy
            'step_penalty': step_penalty,
        },
        'progress': {
            'chip_gain_scale': 1.0,
            'chip_normalization': 1.0,  # No normalization in legacy
            'target_threshold_bonuses': [],
        },
        'hand_quality': {
            'enabled': False,  # Not in legacy
            'bonuses': {},
        },
        'penalties': {
            'invalid_action': 0,
            'desperate_play': 0,
            'desperate_threshold': 0.5,
        },
        'discard_shaping': {
            'enabled': False,  # Disabled in legacy
            'smart_discard_bonus': 0,
            'draw_chase_bonus': 0,
            'weak_hand_play_penalty': 0,
            'weak_hand_threshold': 2,
            'hand_improvement_scale': 0,
            'hand_improvement_max': 0,
            'doomed_play_penalty': 0,
            'valuable_hand_discard_penalty': 0,
            'valuable_hand_threshold': 4,
            'unused_discards_penalty': 0,
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
        'version': 'legacy',
        'description': 'Legacy reward structure for backwards compatibility',
    }
    return RewardShaper(config=legacy_config)
