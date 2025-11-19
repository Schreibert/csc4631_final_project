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
    Balatro single-blind poker environment with batched stepping.

    Observation Space:
        Box(8,) containing:
        - plays_left (0-4)
        - discards_left (0-3)
        - chips_to_target (0-inf)
        - has_pair (0/1)
        - has_trips (0/1)
        - straight_potential (0/1)
        - flush_potential (0/1)
        - max_rank_bucket (0-5)

    Action Space:
        Discrete(7) - see core.ACTION_* constants

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

        # Define spaces
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
            high=np.array([4, 3, np.iinfo(np.int32).max, 1, 1, 1, 1, 5], dtype=np.int32),
            dtype=np.int32
        )

        self.action_space = spaces.Discrete(core.NUM_ACTIONS)

        # RNG for episode seeds
        self._np_random = np.random.RandomState(seed)
        self._episode_count = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset environment to new episode"""
        super().reset(seed=seed)

        if seed is not None:
            self._np_random = np.random.RandomState(seed)

        # Generate episode seed
        episode_seed = self._np_random.randint(0, 2**32)
        self._episode_count += 1

        # Reset C++ simulator
        obs_array = self.sim.reset(self.target_score, episode_seed)
        obs = np.array(obs_array, dtype=np.int32)

        return obs, {}

    def step(self, action: int):
        """Execute single action"""
        result = self.sim.step_batch([action])

        # Convert observation
        obs = np.array(result.final_obs, dtype=np.int32)

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
            'chips': self.target_score - obs[core.OBS_CHIPS_TO_TARGET],
        }

        return obs, reward, terminated, truncated, info

    def step_batch(self, actions: list[int]):
        """
        Execute multiple actions in one call (for efficiency).

        Returns:
            obs: final observation
            rewards: list of per-step rewards (with shaping)
            terminated: bool
            truncated: bool (always False)
            info: dict with 'win' and 'chips'
        """
        result = self.sim.step_batch(actions)

        obs = np.array(result.final_obs, dtype=np.int32)

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
            'chips': self.target_score - obs[core.OBS_CHIPS_TO_TARGET],
            'raw_rewards': result.rewards,  # Unmodified rewards
        }

        return obs, rewards, terminated, truncated, info

    def render(self):
        """Not implemented for v0"""
        pass

    def close(self):
        """Clean up resources"""
        pass
