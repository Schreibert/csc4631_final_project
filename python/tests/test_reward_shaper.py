"""
Unit tests for reward_shaper module.

Tests reward calculation logic, config loading, and edge cases.
"""

import pytest
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from balatro_env.reward_shaper import RewardShaper, create_legacy_reward_shaper


class TestRewardShaperInit:
    """Test RewardShaper initialization and config loading"""

    def test_load_default_config(self):
        """Test loading default rewards_config.yaml"""
        shaper = RewardShaper()
        assert shaper.config is not None
        assert 'outcome' in shaper.config
        assert 'efficiency' in shaper.config

    def test_load_custom_dict_config(self):
        """Test loading config from dict"""
        custom_config = {
            'outcome': {'win_bonus': 2000, 'loss_penalty': 1000},
            'efficiency': {'play_conservation_bonus': 300, 'step_penalty': 2.0},
            'progress': {
                'chip_gain_scale': 1.5,
                'chip_normalization': 200,
                'target_threshold_bonuses': [],
            },
            'hand_quality': {'enabled': False, 'bonuses': {}},
            'penalties': {
                'invalid_action': 30,
                'desperate_play': 15,
                'desperate_threshold': 0.4,
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
        }

        shaper = RewardShaper(config=custom_config)
        assert shaper.config['outcome']['win_bonus'] == 2000
        assert shaper.config['efficiency']['step_penalty'] == 2.0

    def test_load_yaml_file(self):
        """Test loading config from YAML file"""
        # Create temporary YAML file
        yaml_content = """
outcome:
  win_bonus: 1500
  loss_penalty: 750
efficiency:
  play_conservation_bonus: 250
  step_penalty: 1.5
progress:
  chip_gain_scale: 1.0
  chip_normalization: 100
  target_threshold_bonuses: []
hand_quality:
  enabled: true
  bonuses:
    pair: 5
penalties:
  invalid_action: 20
  desperate_play: 10
  desperate_threshold: 0.5
advanced:
  safety_margin_bonus:
    enabled: false
    per_chip_over_target: 0
    max_bonus: 0
  exploration:
    enabled: false
    action_diversity_bonus: 0
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            shaper = RewardShaper(config_path=temp_path)
            assert shaper.config['outcome']['win_bonus'] == 1500
            assert shaper.config['efficiency']['play_conservation_bonus'] == 250
        finally:
            os.unlink(temp_path)

    def test_missing_required_keys(self):
        """Test that missing required keys raise ValueError"""
        invalid_config = {
            'outcome': {'win_bonus': 1000, 'loss_penalty': 500},
            # Missing other required keys
        }

        with pytest.raises(ValueError, match="Missing required config key"):
            RewardShaper(config=invalid_config)


class TestRewardShaping:
    """Test reward shaping calculations"""

    @pytest.fixture
    def simple_shaper(self):
        """Create a shaper with simple config for testing (hand quality disabled by default)"""
        config = {
            'outcome': {'win_bonus': 1000, 'loss_penalty': 500},
            'efficiency': {'play_conservation_bonus': 200, 'step_penalty': 1.0},
            'progress': {
                'chip_gain_scale': 1.0,
                'chip_normalization': 100,
                'target_threshold_bonuses': [
                    {'threshold': 0.5, 'bonus': 50},
                    {'threshold': 0.75, 'bonus': 75},
                ],
            },
            'hand_quality': {
                'enabled': False,  # Disabled to isolate other reward components in tests
                'bonuses': {
                    'high_card': 0,
                    'pair': 5,
                    'two_pair': 10,
                    'flush': 20,
                },
            },
            'penalties': {
                'invalid_action': 25,
                'desperate_play': 10,
                'desperate_threshold': 0.5,
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
        }
        shaper = RewardShaper(config=config)
        shaper.reset_episode_state()
        return shaper

    def test_basic_chip_reward(self, simple_shaper):
        """Test basic chip gain reward with normalization"""
        reward = simple_shaper.shape_reward(
            raw_chip_delta=100,  # Gain 100 chips
            current_chips=100,
            target_score=300,
            action=0,
            plays_left=3,
            discards_left=2,
            done=False,
            win=False
        )

        # Expected: (100 * 1.0 / 100) - 1.0 (step penalty) = 1.0 - 1.0 = 0.0
        assert reward == pytest.approx(0.0, abs=0.01)

    def test_step_penalty_only(self, simple_shaper):
        """Test step penalty on zero chip gain"""
        reward = simple_shaper.shape_reward(
            raw_chip_delta=0,  # No chip gain
            current_chips=50,
            target_score=300,
            action=6,  # Discard
            plays_left=3,
            discards_left=2,
            done=False,
            win=False
        )

        # Expected: 0 - 1.0 (step penalty) = -1.0
        assert reward == pytest.approx(-1.0, abs=0.01)

    def test_win_bonus(self, simple_shaper):
        """Test win bonus with play conservation"""
        reward = simple_shaper.shape_reward(
            raw_chip_delta=50,
            current_chips=300,  # Met target
            target_score=300,
            action=0,
            plays_left=2,  # 2 plays remaining
            discards_left=1,
            done=True,
            win=True
        )

        # Expected: (50 / 100) - 1.0 + 1000 (win) + (2 * 200) (conservation)
        # = 0.5 - 1.0 + 1000 + 400 = 1399.5
        assert reward == pytest.approx(1399.5, abs=0.01)

    def test_loss_penalty(self, simple_shaper):
        """Test loss penalty"""
        reward = simple_shaper.shape_reward(
            raw_chip_delta=0,
            current_chips=250,  # Below target
            target_score=300,
            action=0,
            plays_left=0,  # No plays left
            discards_left=0,
            done=True,
            win=False
        )

        # Expected: 0 - 1.0 - 500 (loss penalty) = -501.0
        assert reward == pytest.approx(-501.0, abs=0.01)

    def test_threshold_bonus_single(self, simple_shaper):
        """Test threshold crossing bonus"""
        # Cross 50% threshold (150 chips for target 300)
        reward = simple_shaper.shape_reward(
            raw_chip_delta=100,
            current_chips=150,  # Just crossed 50%
            target_score=300,
            action=0,
            plays_left=3,
            discards_left=2,
            done=False,
            win=False
        )

        # Expected: (100 / 100) - 1.0 + 50 (threshold bonus) = 1.0 - 1.0 + 50 = 50.0
        assert reward == pytest.approx(50.0, abs=0.01)

    def test_threshold_bonus_not_repeated(self, simple_shaper):
        """Test that threshold bonus only awarded once"""
        # Cross 50% threshold first time
        reward1 = simple_shaper.shape_reward(
            raw_chip_delta=100,
            current_chips=150,
            target_score=300,
            action=0,
            plays_left=3,
            discards_left=2,
            done=False,
            win=False
        )

        # Should get bonus
        assert reward1 == pytest.approx(50.0, abs=0.01)

        # Try again at 160 chips (still above 50%)
        reward2 = simple_shaper.shape_reward(
            raw_chip_delta=10,
            current_chips=160,
            target_score=300,
            action=0,
            plays_left=3,
            discards_left=2,
            done=False,
            win=False
        )

        # Should NOT get bonus again
        # Expected: (10 / 100) - 1.0 = 0.1 - 1.0 = -0.9
        assert reward2 == pytest.approx(-0.9, abs=0.01)

    def test_hand_quality_bonus(self):
        """Test hand quality bonus"""
        # Create shaper with hand quality enabled
        config = {
            'outcome': {'win_bonus': 1000, 'loss_penalty': 500},
            'efficiency': {'play_conservation_bonus': 0, 'step_penalty': 1.0},
            'progress': {
                'chip_gain_scale': 1.0,
                'chip_normalization': 100,
                'target_threshold_bonuses': [],
            },
            'hand_quality': {
                'enabled': True,  # Enabled for this test
                'bonuses': {
                    'pair': 5,
                },
            },
            'penalties': {
                'invalid_action': 0,
                'desperate_play': 0,
                'desperate_threshold': 0.5,
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
        }
        shaper = RewardShaper(config=config)
        shaper.reset_episode_state()

        reward = shaper.shape_reward(
            raw_chip_delta=80,
            current_chips=80,
            target_score=300,
            action=1,  # Action 1 = play pair
            plays_left=3,
            discards_left=2,
            done=False,
            win=False,
            hand_type='pair'
        )

        # Expected: (80 / 100) - 1.0 + 5 (pair bonus) = 0.8 - 1.0 + 5 = 4.8
        assert reward == pytest.approx(4.8, abs=0.01)

    def test_desperate_play_penalty(self, simple_shaper):
        """Test desperate play penalty"""
        reward = simple_shaper.shape_reward(
            raw_chip_delta=20,
            current_chips=100,  # 100/300 = 33%, below 50% threshold
            target_score=300,
            action=4,  # Action 4 = play high card
            plays_left=1,  # Low on plays
            discards_left=0,
            done=False,
            win=False
        )

        # Expected: (20 / 100) - 1.0 - 10 (desperate penalty) = 0.2 - 1.0 - 10 = -10.8
        assert reward == pytest.approx(-10.8, abs=0.01)


class TestResetEpisodeState:
    """Test episode state tracking"""

    def test_reset_clears_thresholds(self):
        """Test that reset clears threshold tracking"""
        config = {
            'outcome': {'win_bonus': 1000, 'loss_penalty': 500},
            'efficiency': {'play_conservation_bonus': 0, 'step_penalty': 1.0},
            'progress': {
                'chip_gain_scale': 1.0,
                'chip_normalization': 100,
                'target_threshold_bonuses': [
                    {'threshold': 0.5, 'bonus': 100},
                ],
            },
            'hand_quality': {'enabled': False, 'bonuses': {}},
            'penalties': {
                'invalid_action': 0,
                'desperate_play': 0,
                'desperate_threshold': 0.5,
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
        }

        shaper = RewardShaper(config=config)

        # Episode 1: Cross threshold
        shaper.reset_episode_state()
        reward1 = shaper.shape_reward(
            raw_chip_delta=100,
            current_chips=150,
            target_score=300,
            action=0,
            plays_left=3,
            discards_left=2,
            done=False,
            win=False
        )
        # Should get threshold bonus: (100/100) - 1 + 100 = 100
        assert reward1 == pytest.approx(100.0, abs=0.01)

        # Episode 2: Reset and cross same threshold again
        shaper.reset_episode_state()
        reward2 = shaper.shape_reward(
            raw_chip_delta=100,
            current_chips=150,
            target_score=300,
            action=0,
            plays_left=3,
            discards_left=2,
            done=False,
            win=False
        )
        # Should get threshold bonus AGAIN: (100/100) - 1 + 100 = 100
        assert reward2 == pytest.approx(100.0, abs=0.01)


class TestLegacyRewardShaper:
    """Test backwards compatibility with legacy reward structure"""

    def test_legacy_shaper_creation(self):
        """Test creating legacy shaper from old parameters"""
        shaper = create_legacy_reward_shaper(
            win_bonus=1500,
            loss_penalty=750,
            step_penalty=2.0
        )

        assert shaper.config['outcome']['win_bonus'] == 1500
        assert shaper.config['outcome']['loss_penalty'] == 750
        assert shaper.config['efficiency']['step_penalty'] == 2.0
        assert shaper.config['efficiency']['play_conservation_bonus'] == 0
        assert shaper.config['hand_quality']['enabled'] is False

    def test_legacy_shaper_behavior(self):
        """Test that legacy shaper matches old reward logic"""
        shaper = create_legacy_reward_shaper(
            win_bonus=1000,
            loss_penalty=500,
            step_penalty=1.0
        )
        shaper.reset_episode_state()

        # Test basic step: chip gain - step penalty
        reward = shaper.shape_reward(
            raw_chip_delta=50,
            current_chips=50,
            target_score=300,
            action=0,
            plays_left=3,
            discards_left=2,
            done=False,
            win=False
        )
        # Legacy: raw_chip_delta - step_penalty = 50 - 1 = 49
        assert reward == pytest.approx(49.0, abs=0.01)

        # Test win: chip gain - step penalty + win bonus
        reward_win = shaper.shape_reward(
            raw_chip_delta=100,
            current_chips=300,
            target_score=300,
            action=0,
            plays_left=2,
            discards_left=1,
            done=True,
            win=True
        )
        # Legacy: 100 - 1 + 1000 = 1099 (no conservation bonus)
        assert reward_win == pytest.approx(1099.0, abs=0.01)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_zero_chip_delta(self):
        """Test handling of zero chip gains"""
        shaper = RewardShaper()
        shaper.reset_episode_state()

        reward = shaper.shape_reward(
            raw_chip_delta=0,
            current_chips=100,
            target_score=300,
            action=6,  # Discard
            plays_left=3,
            discards_left=2,
            done=False,
            win=False
        )

        # Should only have step penalty
        assert reward < 0

    def test_negative_chip_delta(self):
        """Test handling of negative chip deltas (shouldn't happen, but test anyway)"""
        shaper = RewardShaper()
        shaper.reset_episode_state()

        reward = shaper.shape_reward(
            raw_chip_delta=-50,
            current_chips=50,
            target_score=300,
            action=0,
            plays_left=3,
            discards_left=2,
            done=False,
            win=False
        )

        # Negative chip delta gets scaled/normalized too
        assert reward < 0

    def test_very_large_chip_gain(self):
        """Test handling of very large chip gains"""
        shaper = RewardShaper()
        shaper.reset_episode_state()

        reward = shaper.shape_reward(
            raw_chip_delta=10000,
            current_chips=10000,
            target_score=300,
            action=0,
            plays_left=3,
            discards_left=2,
            done=False,
            win=False
        )

        # Should be positive, but normalized
        assert reward > 0

    def test_already_at_target(self):
        """Test reward when already at/above target (should still work)"""
        shaper = RewardShaper()
        shaper.reset_episode_state()

        # Simulate that we were already at 500 chips (so no threshold crossing)
        shaper.previous_chips = 500

        reward = shaper.shape_reward(
            raw_chip_delta=0,
            current_chips=500,  # Already above target, no gain
            target_score=300,
            action=0,
            plays_left=3,
            discards_left=2,
            done=False,
            win=False
        )

        # Should only have step penalty (no chip gain, no threshold bonuses)
        assert reward <= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
