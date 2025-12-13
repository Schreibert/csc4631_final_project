"""
Gymnasium-style environment wrapper for Balatro poker simulator.

This module provides BalatroBatchedSimEnv, a Gymnasium-compatible environment
that wraps the high-performance C++ Balatro poker simulator for RL training.

Features:
    - Dict observation space with 22 features including hand analysis
    - Dict action space with direct card selection (type + 8-card mask)
    - Configurable reward shaping via YAML or constructor parameters
    - Batch action execution for efficiency
    - RL helper methods (best hand, score prediction, action enumeration)

See Also:
    - rewards_config.yaml: Default reward configuration
    - reward_shaper.py: RewardShaper class for custom rewards
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any

try:
    from balatro_env import _balatro_core as core
    from balatro_env.reward_shaper import RewardShaper, create_legacy_reward_shaper
except ImportError:
    import _balatro_core as core
    from reward_shaper import RewardShaper, create_legacy_reward_shaper


class BalatroBatchedSimEnv(gym.Env):
    """
    Balatro single-blind poker environment with direct card control.

    Observation Space:
        Dict with:
        - 'plays_left': Discrete(5) [0-4]
        - 'discards_left': Discrete(4) [0-3]
        - 'chips': Box(1) [current chips scored]
        - 'chips_to_target': Box(1) [chips needed to win]
        - 'deck_remaining': Box(1) [cards left in deck, 0-52]
        - 'num_face_cards': Box(1) [face cards (J,Q,K) in hand, 0-8]
        - 'num_aces': Box(1) [aces in hand, 0-8]
        - 'card_ranks': MultiDiscrete([13] * 8) [rank of each card in hand, 0-12]
        - 'card_suits': MultiDiscrete([4] * 8) [suit of each card in hand, 0-3]
        - 'has_pair': Discrete(2) [0/1]
        - 'has_trips': Discrete(2) [0/1]
        - 'straight_potential': Discrete(2) [0/1]
        - 'flush_potential': Discrete(2) [0/1]

    Action Space:
        Dict with:
        - 'type': Discrete(2) [0=PLAY, 1=DISCARD]
        - 'card_mask': MultiBinary(8) [which cards to play/discard]

    Rewards:
        Configurable via YAML (rewards_config.yaml) or constructor parameters.
        Default structure:
        - Chip gain (normalized and scaled)
        - Win bonus (+1000) with play conservation bonus
        - Loss penalty (-500)
        - Step penalty (-1 per action)
        - Threshold bonuses (75%, 90% of target)
        - Hand quality bonuses (small, encourage strong hands)
        - Penalties for desperate/invalid plays

        See rewards_config.yaml for full configuration options.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        target_score: int = 300,
        win_bonus: int = 1000,
        loss_penalty: int = 500,
        step_penalty: int = 1,
        seed: int | None = None,
        reward_config: Optional[Dict[str, Any]] = None,
        reward_config_path: Optional[str] = None
    ):
        """
        Initialize Balatro poker environment.

        Args:
            target_score: Chips needed to win an episode (default: 300).
                Higher values make episodes harder and longer.
            win_bonus: Legacy reward bonus for winning (default: 1000).
                Only used if no reward_config provided.
            loss_penalty: Legacy penalty for losing (default: 500).
                Only used if no reward_config provided.
            step_penalty: Legacy per-action cost (default: 1).
                Only used if no reward_config provided.
            seed: Random seed for episode generation. If None, uses
                random seed. Same seed produces identical episode sequences.
            reward_config: Dict with reward configuration (overrides YAML).
                See rewards_config.yaml for expected structure.
            reward_config_path: Path to custom reward YAML file.
                If None and reward_config is None, loads default config.

        Notes:
            Reward configuration priority:
            1. If reward_config dict is provided, use it directly
            2. If reward_config_path is provided, load from that file
            3. If legacy params (win_bonus, etc.) differ from defaults, use legacy mode
            4. Otherwise, load default rewards_config.yaml

        Example:
            >>> # Default configuration
            >>> env = BalatroBatchedSimEnv(target_score=300)
            >>>
            >>> # Custom YAML config
            >>> env = BalatroBatchedSimEnv(
            ...     target_score=500,
            ...     reward_config_path='my_rewards.yaml'
            ... )
            >>>
            >>> # Legacy mode (backwards compatible)
            >>> env = BalatroBatchedSimEnv(
            ...     target_score=300,
            ...     win_bonus=100,
            ...     step_penalty=5
            ... )
        """
        super().__init__()

        self.target_score = target_score

        # Create reward shaper (supports both new YAML config and legacy parameters)
        # Check if user is explicitly using legacy mode (passed non-default values)
        using_legacy_params = (
            win_bonus != 1000 or loss_penalty != 500 or step_penalty != 1
        )

        if reward_config is not None or reward_config_path is not None:
            # Explicit YAML config provided
            self.reward_shaper = RewardShaper(config=reward_config, config_path=reward_config_path)
        elif using_legacy_params and reward_config is None and reward_config_path is None:
            # Legacy mode: user passed custom legacy params and no YAML config
            self.reward_shaper = create_legacy_reward_shaper(win_bonus, loss_penalty, step_penalty)
        else:
            # Default: load default YAML config
            self.reward_shaper = RewardShaper()

        # Keep legacy parameters as attributes for backwards compatibility
        self.win_bonus = win_bonus
        self.loss_penalty = loss_penalty
        self.step_penalty = step_penalty

        # Create C++ simulator
        self.sim = core.Simulator()

        # Define observation space (Dict)
        self.observation_space = spaces.Dict({
            'plays_left': spaces.Discrete(5),
            'discards_left': spaces.Discrete(4),
            'chips': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            'chips_to_target': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            'deck_remaining': spaces.Box(low=0, high=52, shape=(1,), dtype=np.int32),
            'num_face_cards': spaces.Box(low=0, high=core.HAND_SIZE, shape=(1,), dtype=np.int32),
            'num_aces': spaces.Box(low=0, high=core.HAND_SIZE, shape=(1,), dtype=np.int32),
            'card_ranks': spaces.MultiDiscrete([core.NUM_RANKS] * core.HAND_SIZE),
            'card_suits': spaces.MultiDiscrete([core.NUM_SUITS] * core.HAND_SIZE),
            'has_pair': spaces.Discrete(2),
            'has_trips': spaces.Discrete(2),
            'straight_potential': spaces.Discrete(2),
            'flush_potential': spaces.Discrete(2),
            # Best hand analysis (for RL agents)
            'best_hand_type': spaces.Discrete(9),  # 0-8 (HandType enum)
            'best_hand_score': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            # Complete hand pattern flags
            'has_two_pair': spaces.Discrete(2),
            'has_full_house': spaces.Discrete(2),
            'has_four_of_kind': spaces.Discrete(2),
            'has_straight': spaces.Discrete(2),
            'has_flush': spaces.Discrete(2),
            'has_straight_flush': spaces.Discrete(2),
        })

        # Define action space (Dict)
        self.action_space = spaces.Dict({
            'type': spaces.Discrete(2),  # 0=PLAY, 1=DISCARD
            'card_mask': spaces.MultiBinary(core.HAND_SIZE)
        })

        # RNG for episode seeds and random actions
        self._np_random = np.random.RandomState(seed)
        self._episode_count = 0
        self._last_obs = None

    def _obs_to_dict(self, obs):
        """Convert C++ Observation to gym Dict observation"""
        return {
            'plays_left': obs.plays_left,
            'discards_left': obs.discards_left,
            'chips': np.array([obs.chips], dtype=np.int32),
            'chips_to_target': np.array([obs.chips_to_target], dtype=np.int32),
            'deck_remaining': np.array([obs.deck_remaining], dtype=np.int32),
            'num_face_cards': np.array([obs.num_face_cards], dtype=np.int32),
            'num_aces': np.array([obs.num_aces], dtype=np.int32),
            'card_ranks': obs.card_ranks,  # Already a numpy array from C++
            'card_suits': obs.card_suits,  # Already a numpy array from C++
            'has_pair': obs.has_pair,
            'has_trips': obs.has_trips,
            'straight_potential': obs.straight_potential,
            'flush_potential': obs.flush_potential,
            # Best hand analysis
            'best_hand_type': obs.best_hand_type,
            'best_hand_score': np.array([obs.best_hand_score], dtype=np.int32),
            # Complete hand pattern flags
            'has_two_pair': obs.has_two_pair,
            'has_full_house': obs.has_full_house,
            'has_four_of_kind': obs.has_four_of_kind,
            'has_straight': obs.has_straight,
            'has_flush': obs.has_flush,
            'has_straight_flush': obs.has_straight_flush,
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset environment to new episode"""
        super().reset(seed=seed)

        if seed is not None:
            self._np_random = np.random.RandomState(seed)

        # Generate episode seed (use 2**31 - 1 for int32 compatibility)
        episode_seed = self._np_random.randint(0, 2**31 - 1)
        self._episode_count += 1

        # Reset C++ simulator
        obs = self.sim.reset(self.target_score, episode_seed)
        self._last_obs = obs

        # Reset reward shaper episode state
        self.reward_shaper.reset_episode_state()

        return self._obs_to_dict(obs), {}

    def step(self, action: dict):
        """Execute single action

        Args:
            action: Dict with 'type' (0=PLAY, 1=DISCARD) and 'card_mask' (bool array)
        """
        # Capture state BEFORE action for discard shaping
        obs_before = self._last_obs
        best_hand_type_before = obs_before.best_hand_type if obs_before else 0
        flush_potential_before = obs_before.flush_potential if obs_before else 0
        straight_potential_before = obs_before.straight_potential if obs_before else 0
        discards_left_before = obs_before.discards_left if obs_before else 0

        # Detect if action is a discard
        is_discard_action = (action['type'] == 1)

        # Convert gym action dict to C++ Action
        cpp_action = core.Action()
        cpp_action.type = core.PLAY if action['type'] == 0 else core.DISCARD
        cpp_action.card_mask = [bool(x) for x in action['card_mask']]

        # Execute action
        result = self.sim.step_batch([cpp_action])

        # Convert observation
        obs_dict = self._obs_to_dict(result.final_obs)
        self._last_obs = result.final_obs

        # Calculate reward using reward shaper
        # Convert action dict to simple action code (for hand type inference)
        # For now, we don't have direct action codes, so we'll use a placeholder
        action_code = 0 if action['type'] == 0 else 6  # PLAY=0, DISCARD=6

        reward = self.reward_shaper.shape_reward(
            raw_chip_delta=result.rewards[0],
            current_chips=result.final_obs.chips,
            target_score=self.target_score,
            action=action_code,
            plays_left=result.final_obs.plays_left,
            discards_left=result.final_obs.discards_left,
            done=result.done,
            win=result.win,
            hand_type=None,  # We don't have hand type info from C++ yet
            # New discard shaping parameters
            best_hand_type=result.final_obs.best_hand_type,
            flush_potential=(flush_potential_before == 1),
            straight_potential=(straight_potential_before == 1),
            is_discard_action=is_discard_action,
            discards_left_before=discards_left_before
        )

        terminated = result.done
        truncated = False

        info = {
            'win': result.win,
            'chips': result.final_obs.chips,
            'raw_reward': result.rewards[0],  # Include raw reward for debugging
        }

        return obs_dict, reward, terminated, truncated, info

    def step_batch(self, actions: list[dict]):
        """
        Execute multiple actions in one call (for efficiency).

        Args:
            actions: List of action dicts, each with 'type' and 'card_mask'

        Returns:
            obs: final observation (dict)
            rewards: list of per-step rewards (with shaping)
            terminated: bool
            truncated: bool (always False)
            info: dict with 'win' and 'chips'

        Note: In batch mode, threshold bonuses are approximated since we don't
        have intermediate observations. For precise threshold tracking, use
        single-step mode.
        """
        # Convert gym actions to C++ Actions
        cpp_actions = []
        for action in actions:
            cpp_action = core.Action()
            cpp_action.type = core.PLAY if action['type'] == 0 else core.DISCARD
            cpp_action.card_mask = [bool(x) for x in action['card_mask']]
            cpp_actions.append(cpp_action)

        result = self.sim.step_batch(cpp_actions)

        obs_dict = self._obs_to_dict(result.final_obs)
        self._last_obs = result.final_obs

        # Apply reward shaping to each step
        # Note: We approximate chip progression since we only have final observation
        shaped_rewards = []
        cumulative_chips = self.reward_shaper.previous_chips  # Start from previous episode state

        for i, (raw_reward, action) in enumerate(zip(result.rewards, actions)):
            # Estimate chips at this step
            cumulative_chips += raw_reward
            is_last_step = (i == len(actions) - 1)
            is_done = result.done and is_last_step

            # Convert action dict to action code
            action_code = 0 if action['type'] == 0 else 6  # PLAY=0, DISCARD=6

            # For batch mode, we approximate plays_left/discards_left
            # (exact tracking would require intermediate observations)
            approx_plays_left = result.final_obs.plays_left
            approx_discards_left = result.final_obs.discards_left

            shaped_reward = self.reward_shaper.shape_reward(
                raw_chip_delta=raw_reward,
                current_chips=cumulative_chips if is_last_step else cumulative_chips,
                target_score=self.target_score,
                action=action_code,
                plays_left=approx_plays_left,
                discards_left=approx_discards_left,
                done=is_done,
                win=result.win if is_done else False,
                hand_type=None
            )
            shaped_rewards.append(shaped_reward)

        terminated = result.done
        truncated = False

        info = {
            'win': result.win,
            'chips': result.final_obs.chips,
            'raw_rewards': result.rewards,  # Unmodified rewards for debugging
        }

        return obs_dict, shaped_rewards, terminated, truncated, info

    def render(self):
        """Not implemented for v0"""
        pass

    def close(self):
        """Clean up resources"""
        pass

    # RL helper methods (expose C++ functionality)

    def get_best_hand(self):
        """
        Get the best possible poker hand from current state.

        Returns:
            HandEvaluation object with type and rank_sum
        """
        return self.sim.get_best_hand()

    def predict_score(self, action: dict) -> int:
        """
        Predict the score for a PLAY action without executing it.

        Args:
            action: Action dict with 'type' and 'card_mask'

        Returns:
            Predicted chip score (0 if action is DISCARD)
        """
        if action['type'] != 0:  # Not a PLAY action
            return 0

        # Convert to bool array
        card_mask = [bool(x) for x in action['card_mask']]
        return self.sim.predict_play_score(card_mask)

    def get_valid_actions_with_scores(self):
        """
        Enumerate all valid actions with predicted outcomes.

        Returns:
            List of ActionOutcome objects with:
                - action: Action dict
                - valid: bool
                - predicted_chips: int
                - predicted_hand_type: int (HandType enum value)
        """
        cpp_outcomes = self.sim.enumerate_all_actions()

        # Convert C++ ActionOutcome objects to Python dicts
        outcomes = []
        for cpp_outcome in cpp_outcomes:
            outcome = {
                'action': {
                    'type': 0 if cpp_outcome.action.type == core.PLAY else 1,
                    'card_mask': np.array(cpp_outcome.action.card_mask, dtype=np.int8)
                },
                'valid': cpp_outcome.valid,
                'predicted_chips': cpp_outcome.predicted_chips,
                'predicted_hand_type': cpp_outcome.predicted_hand_type
            }
            outcomes.append(outcome)

        return outcomes
