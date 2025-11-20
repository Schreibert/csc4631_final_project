"""Gymnasium-style environment wrapper for Balatro simulator"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from balatro_env import _balatro_core as core
except ImportError:
    import _balatro_core as core


class BalatroBatchedSimEnv(gym.Env):
    """
    Balatro single-blind poker environment with direct card control.

    Observation Space:
        Dict with:
        - 'plays_left': Discrete(5) [0-4]
        - 'discards_left': Discrete(4) [0-3]
        - 'chips': Box(1) [current chips scored]
        - 'chips_to_target': Box(1) [chips needed to win]
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
        - Chip delta for plays
        - +1000 bonus on win
        - -500 penalty on loss
        - -1 per step (optional)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        target_score: int = 300,
        win_bonus: int = 1000,
        loss_penalty: int = 500,
        step_penalty: int = 1,
        seed: int | None = None
    ):
        super().__init__()

        self.target_score = target_score
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
            'card_ranks': spaces.MultiDiscrete([core.NUM_RANKS] * core.HAND_SIZE),
            'card_suits': spaces.MultiDiscrete([core.NUM_SUITS] * core.HAND_SIZE),
            'has_pair': spaces.Discrete(2),
            'has_trips': spaces.Discrete(2),
            'straight_potential': spaces.Discrete(2),
            'flush_potential': spaces.Discrete(2),
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
            'card_ranks': np.array(obs.card_ranks, dtype=np.int32),
            'card_suits': np.array(obs.card_suits, dtype=np.int32),
            'has_pair': obs.has_pair,
            'has_trips': obs.has_trips,
            'straight_potential': obs.straight_potential,
            'flush_potential': obs.flush_potential,
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

        return self._obs_to_dict(obs), {}

    def step(self, action: dict):
        """Execute single action

        Args:
            action: Dict with 'type' (0=PLAY, 1=DISCARD) and 'card_mask' (bool array)
        """
        # Convert gym action dict to C++ Action
        cpp_action = core.Action()
        cpp_action.type = core.PLAY if action['type'] == 0 else core.DISCARD
        cpp_action.card_mask = [bool(x) for x in action['card_mask']]

        # Execute action
        result = self.sim.step_batch([cpp_action])

        # Convert observation
        obs_dict = self._obs_to_dict(result.final_obs)
        self._last_obs = result.final_obs

        # Calculate reward with shaping
        reward = result.rewards[0] - self.step_penalty

        # Add terminal bonuses
        if result.done:
            if result.win:
                reward += self.win_bonus
            else:
                reward -= self.loss_penalty

        terminated = result.done
        truncated = False

        info = {
            'win': result.win,
            'chips': result.final_obs.chips,
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

        # Apply reward shaping to all steps
        rewards = [r - self.step_penalty for r in result.rewards]

        # Add terminal bonus to final step if done
        if result.done and rewards:
            if result.win:
                rewards[-1] += self.win_bonus
            else:
                rewards[-1] -= self.loss_penalty

        terminated = result.done
        truncated = False

        info = {
            'win': result.win,
            'chips': result.final_obs.chips,
            'raw_rewards': result.rewards,  # Unmodified rewards
        }

        return obs_dict, rewards, terminated, truncated, info

    def render(self):
        """Not implemented for v0"""
        pass

    def close(self):
        """Clean up resources"""
        pass
