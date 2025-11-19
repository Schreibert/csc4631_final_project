"""Tests for determinism and interface contract compliance"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import core module directly (gymnasium not required for these tests)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'balatro_env'))
import _balatro_core as core


def test_deterministic_reset():
    """Same seed should produce same initial state"""
    sim1 = core.Simulator()
    sim2 = core.Simulator()

    obs1 = sim1.reset(300, 12345)
    obs2 = sim2.reset(300, 12345)

    assert list(obs1) == list(obs2), "Same seed should give same initial observation"


def test_deterministic_trajectory():
    """Same actions with same seed should give same trajectory"""
    sim1 = core.Simulator()
    sim2 = core.Simulator()

    sim1.reset(500, 99999)
    sim2.reset(500, 99999)

    actions = [core.ACTION_PLAY_BEST, core.ACTION_DISCARD_LOWEST_3, core.ACTION_PLAY_PAIR]

    result1 = sim1.step_batch(actions)
    result2 = sim2.step_batch(actions)

    assert list(result1.final_obs) == list(result2.final_obs)
    assert list(result1.rewards) == list(result2.rewards)
    assert result1.done == result2.done
    assert result1.win == result2.win


def test_batch_vs_sequential():
    """Batched steps should match sequential steps"""
    sim_batch = core.Simulator()
    sim_seq = core.Simulator()

    sim_batch.reset(1000, 55555)
    sim_seq.reset(1000, 55555)

    actions = [core.ACTION_PLAY_BEST, core.ACTION_DISCARD_LOWEST_3]

    # Batch execution
    batch_result = sim_batch.step_batch(actions)

    # Sequential execution
    seq_rewards = []
    for action in actions:
        result = sim_seq.step_batch([action])
        seq_rewards.extend(result.rewards)

    seq_obs = sim_seq.step_batch([]).final_obs  # Get current obs

    # Compare
    assert list(batch_result.final_obs) == list(seq_obs)
    assert list(batch_result.rewards) == seq_rewards


def test_observation_size():
    """Observation should always be size 8"""
    sim = core.Simulator()
    obs = sim.reset(200, 11111)

    assert len(obs) == core.OBS_SIZE
    assert core.OBS_SIZE == 8


def test_action_space():
    """All action codes should work"""
    sim = core.Simulator()

    all_actions = [
        core.ACTION_PLAY_BEST,
        core.ACTION_PLAY_PAIR,
        core.ACTION_DISCARD_NON_PAIRED,
        core.ACTION_DISCARD_LOWEST_3,
        core.ACTION_PLAY_HIGHEST_3,
        core.ACTION_RANDOM_PLAY,
        core.ACTION_RANDOM_DISCARD,
    ]

    for action in all_actions:
        sim.reset(300, 77777)
        result = sim.step_batch([action])
        assert len(result.rewards) == 1


def test_episode_termination():
    """Episode should terminate on win or loss"""
    # Test win
    sim_win = core.Simulator()
    sim_win.reset(10, 11111)  # Very low target

    for _ in range(10):
        result = sim_win.step_batch([core.ACTION_PLAY_BEST])
        if result.done:
            assert result.win, "Low target should result in win"
            break
    else:
        assert False, "Should have won with low target"

    # Test loss
    sim_loss = core.Simulator()
    sim_loss.reset(100000, 22222)  # Very high target

    # Exhaust all 4 plays
    result = sim_loss.step_batch([
        core.ACTION_PLAY_BEST,
        core.ACTION_PLAY_BEST,
        core.ACTION_PLAY_BEST,
        core.ACTION_PLAY_BEST
    ])

    assert result.done, "Should be done after exhausting plays"
    assert not result.win, "Should lose with impossible target"


def test_padding_after_done():
    """Actions after done should be no-ops"""
    sim = core.Simulator()
    sim.reset(10, 33333)

    # Win quickly
    result1 = sim.step_batch([core.ACTION_PLAY_BEST])
    if not result1.done:
        result1 = sim.step_batch([core.ACTION_PLAY_BEST])

    assert result1.done, "Should be done"

    # Try more actions
    result2 = sim.step_batch([core.ACTION_PLAY_BEST, core.ACTION_PLAY_BEST])

    assert all(r == 0 for r in result2.rewards), "Rewards after done should be 0"
    assert result2.done, "Should still be done"


if __name__ == "__main__":
    test_deterministic_reset()
    print("[PASS] Deterministic reset")

    test_deterministic_trajectory()
    print("[PASS] Deterministic trajectory")

    test_batch_vs_sequential()
    print("[PASS] Batch vs sequential equivalence")

    test_observation_size()
    print("[PASS] Observation size")

    test_action_space()
    print("[PASS] Action space")

    test_episode_termination()
    print("[PASS] Episode termination")

    test_padding_after_done()
    print("[PASS] Padding after done")

    print("\nAll tests passed!")
