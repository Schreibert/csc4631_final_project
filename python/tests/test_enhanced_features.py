"""
Integration test for enhanced C++ RL features.

Tests new observation fields, score prediction, and action enumeration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from balatro_env import BalatroBatchedSimEnv
import numpy as np


def test_enhanced_observation_fields():
    """Test that all new observation fields are populated correctly."""
    env = BalatroBatchedSimEnv(target_score=300)
    obs, _ = env.reset(seed=42)

    # Check new best hand fields exist and are populated
    assert 'best_hand_type' in obs, "Missing best_hand_type"
    assert 'best_hand_score' in obs, "Missing best_hand_score"

    # Check complete hand pattern flags
    assert 'has_two_pair' in obs, "Missing has_two_pair"
    assert 'has_full_house' in obs, "Missing has_full_house"
    assert 'has_four_of_kind' in obs, "Missing has_four_of_kind"
    assert 'has_straight' in obs, "Missing has_straight"
    assert 'has_flush' in obs, "Missing has_flush"
    assert 'has_straight_flush' in obs, "Missing has_straight_flush"

    # Verify types
    assert isinstance(obs['best_hand_type'], (int, np.integer)), "best_hand_type not int"
    assert isinstance(obs['best_hand_score'], np.ndarray), "best_hand_score not array"
    assert 0 <= obs['best_hand_type'] <= 8, "best_hand_type out of range"
    assert obs['best_hand_score'][0] >= 0, "best_hand_score negative"

    print("[PASS] Enhanced observation fields test")


def test_score_prediction():
    """Test score prediction without executing actions."""
    env = BalatroBatchedSimEnv(target_score=300)
    obs, _ = env.reset(seed=42)

    # Create a PLAY action (play all cards)
    play_action = {
        'type': 0,  # PLAY
        'card_mask': np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=np.int8)
    }

    # Predict score
    predicted_score = env.predict_score(play_action)
    assert isinstance(predicted_score, (int, np.integer)), "predicted_score not int"
    assert predicted_score >= 0, "predicted_score negative"

    # Execute action and compare
    obs2, reward, done, _, info = env.step(play_action)
    actual_chips_gained = info['raw_reward']

    print(f"  Predicted: {predicted_score}, Actual: {actual_chips_gained}")
    assert predicted_score == actual_chips_gained, "Score prediction mismatch!"

    print("[PASS] Score prediction test")


def test_action_enumeration():
    """Test action enumeration with predicted outcomes."""
    env = BalatroBatchedSimEnv(target_score=300)
    obs, _ = env.reset(seed=42)

    # Get all valid actions
    outcomes = env.get_valid_actions_with_scores()

    assert len(outcomes) > 0, "No actions enumerated"
    print(f"  Enumerated {len(outcomes)} valid actions")

    # Check structure
    for outcome in outcomes[:5]:  # Check first 5
        assert 'action' in outcome, "Missing action field"
        assert 'valid' in outcome, "Missing valid field"
        assert 'predicted_chips' in outcome, "Missing predicted_chips"
        assert 'predicted_hand_type' in outcome, "Missing predicted_hand_type"

        assert outcome['valid'] == True, "Invalid action in enumeration"

    # Verify actions are sorted by score (PLAY actions)
    play_outcomes = [o for o in outcomes if o['action']['type'] == 0]
    if len(play_outcomes) > 1:
        for i in range(len(play_outcomes) - 1):
            assert play_outcomes[i]['predicted_chips'] >= play_outcomes[i+1]['predicted_chips'], \
                "Actions not sorted by predicted score"

    print(f"  Best action predicts {play_outcomes[0]['predicted_chips']} chips")
    print("[PASS] Action enumeration test")


def test_consistency_between_methods():
    """Test that C++ methods are consistent with each other."""
    env = BalatroBatchedSimEnv(target_score=300)
    obs, _ = env.reset(seed=42)

    # Get best hand from observation (best 5-card hand)
    obs_best_score = obs['best_hand_score'][0]

    # Get best hand via helper method
    best_hand = env.get_best_hand()

    # Get top action from enumeration (considers all 1-5 card plays)
    outcomes = env.get_valid_actions_with_scores()
    play_outcomes = [o for o in outcomes if o['action']['type'] == 0]
    if play_outcomes:
        top_action_score = play_outcomes[0]['predicted_chips']

        print(f"  Observation best_hand_score (5 cards): {obs_best_score}")
        print(f"  get_best_hand() type: {best_hand.type}")
        print(f"  Top enumerated action score (1-5 cards): {top_action_score}")

        # Top action should be >= best 5-card hand (might play fewer cards optimally)
        assert top_action_score >= obs_best_score, \
            f"Top action worse than best 5-card hand: {top_action_score} < {obs_best_score}"

    print("[PASS] Consistency test (top action >= best 5-card hand)")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Enhanced RL Features")
    print("=" * 60)

    test_enhanced_observation_fields()
    test_score_prediction()
    test_action_enumeration()
    test_consistency_between_methods()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
