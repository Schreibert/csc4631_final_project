#include <gtest/gtest.h>
#include "balatro/simulator.hpp"

using namespace balatro;

TEST(SimulatorTest, Reset) {
    Simulator sim;
    auto obs = sim.reset(300, 12345);

    // Check initial state
    EXPECT_EQ(obs[OBS_PLAYS_LEFT], 4);
    EXPECT_EQ(obs[OBS_DISCARDS_LEFT], 3);
    EXPECT_EQ(obs[OBS_CHIPS_TO_TARGET], 300);

    EXPECT_FALSE(sim.state_view().is_done());
    EXPECT_FALSE(sim.state_view().is_win());
}

TEST(SimulatorTest, DeterministicReset) {
    Simulator sim1, sim2;

    auto obs1 = sim1.reset(300, 99999);
    auto obs2 = sim2.reset(300, 99999);

    // Same seed should produce same initial observation
    for (int i = 0; i < OBS_SIZE; ++i) {
        EXPECT_EQ(obs1[i], obs2[i]);
    }

    // Same initial hand
    const auto& hand1 = sim1.state_view().get_hand();
    const auto& hand2 = sim2.state_view().get_hand();
    for (int i = 0; i < HAND_SIZE; ++i) {
        EXPECT_EQ(hand1[i], hand2[i]);
    }
}

TEST(SimulatorTest, StepBatch) {
    Simulator sim;
    sim.reset(5000, 54321); // High target so we don't win immediately

    // Execute some actions
    std::vector<int> actions = {ACTION_PLAY_BEST, ACTION_DISCARD_LOWEST_3};
    auto result = sim.step_batch(actions);

    EXPECT_EQ(result.rewards.size(), 2);
    EXPECT_GE(result.rewards[0], 0); // Play should give some reward
    EXPECT_EQ(result.rewards[1], 0); // Discard gives no reward

    // Should have used resources (if not done)
    if (!result.done) {
        EXPECT_EQ(sim.state_view().get_plays_left(), 3);
        EXPECT_EQ(sim.state_view().get_discards_left(), 2);
    }
}

TEST(SimulatorTest, EpisodeTerminationWin) {
    Simulator sim;
    sim.reset(10, 11111); // Very low target, should win easily

    // Keep playing until done
    int max_steps = 10;
    bool won = false;

    for (int i = 0; i < max_steps; ++i) {
        if (sim.state_view().is_done()) {
            won = sim.state_view().is_win();
            break;
        }

        auto result = sim.step_batch({ACTION_PLAY_BEST});
        if (result.done) {
            won = result.win;
            break;
        }
    }

    EXPECT_TRUE(won); // Should win with such a low target
}

TEST(SimulatorTest, EpisodeTerminationLoss) {
    Simulator sim;
    sim.reset(10000, 22222); // Very high target, should lose

    // Exhaust all plays
    std::vector<int> actions = {
        ACTION_PLAY_BEST,
        ACTION_PLAY_BEST,
        ACTION_PLAY_BEST,
        ACTION_PLAY_BEST
    };

    auto result = sim.step_batch(actions);

    EXPECT_TRUE(result.done);
    EXPECT_FALSE(result.win); // Should lose
    EXPECT_EQ(sim.state_view().get_plays_left(), 0);
}

TEST(SimulatorTest, DeterministicTrajectory) {
    Simulator sim1, sim2;

    sim1.reset(200, 77777);
    sim2.reset(200, 77777);

    std::vector<int> actions = {
        ACTION_PLAY_BEST,
        ACTION_DISCARD_LOWEST_3,
        ACTION_PLAY_PAIR,
        ACTION_PLAY_BEST
    };

    auto result1 = sim1.step_batch(actions);
    auto result2 = sim2.step_batch(actions);

    // Same actions, same seed should give same results
    EXPECT_EQ(result1.rewards.size(), result2.rewards.size());
    for (size_t i = 0; i < result1.rewards.size(); ++i) {
        EXPECT_EQ(result1.rewards[i], result2.rewards[i]);
    }

    EXPECT_EQ(result1.done, result2.done);
    EXPECT_EQ(result1.win, result2.win);

    for (int i = 0; i < OBS_SIZE; ++i) {
        EXPECT_EQ(result1.final_obs[i], result2.final_obs[i]);
    }
}

TEST(SimulatorTest, PaddingAfterDone) {
    Simulator sim;
    sim.reset(10, 33333); // Low target

    // Win quickly
    auto result1 = sim.step_batch({ACTION_PLAY_BEST});

    if (!result1.done) {
        // If not done yet, play once more
        result1 = sim.step_batch({ACTION_PLAY_BEST});
    }

    ASSERT_TRUE(sim.state_view().is_done());

    // Now execute more actions - should be padded with 0 rewards
    auto result2 = sim.step_batch({ACTION_PLAY_BEST, ACTION_PLAY_BEST});

    EXPECT_EQ(result2.rewards.size(), 2);
    EXPECT_EQ(result2.rewards[0], 0);
    EXPECT_EQ(result2.rewards[1], 0);
}

TEST(SimulatorTest, ObservationValidity) {
    Simulator sim;
    auto obs = sim.reset(500, 44444);

    // Check observation ranges
    EXPECT_GE(obs[OBS_PLAYS_LEFT], 0);
    EXPECT_LE(obs[OBS_PLAYS_LEFT], 4);
    EXPECT_GE(obs[OBS_DISCARDS_LEFT], 0);
    EXPECT_LE(obs[OBS_DISCARDS_LEFT], 3);
    EXPECT_GE(obs[OBS_CHIPS_TO_TARGET], 0);
    EXPECT_GE(obs[OBS_HAS_PAIR], 0);
    EXPECT_LE(obs[OBS_HAS_PAIR], 1);
    EXPECT_GE(obs[OBS_HAS_TRIPS], 0);
    EXPECT_LE(obs[OBS_HAS_TRIPS], 1);
    EXPECT_GE(obs[OBS_STRAIGHT_POT], 0);
    EXPECT_LE(obs[OBS_STRAIGHT_POT], 1);
    EXPECT_GE(obs[OBS_FLUSH_POT], 0);
    EXPECT_LE(obs[OBS_FLUSH_POT], 1);
    EXPECT_GE(obs[OBS_MAX_RANK_BUCKET], 0);
    EXPECT_LE(obs[OBS_MAX_RANK_BUCKET], 5);
}

TEST(SimulatorTest, AllActions) {
    Simulator sim;
    sim.reset(300, 55555);

    // Test each action doesn't crash
    std::vector<int> all_actions = {
        ACTION_PLAY_BEST,
        ACTION_PLAY_PAIR,
        ACTION_DISCARD_NON_PAIRED,
        ACTION_DISCARD_LOWEST_3,
        ACTION_PLAY_HIGHEST_3,
        ACTION_RANDOM_PLAY,
        ACTION_RANDOM_DISCARD
    };

    for (int action : all_actions) {
        Simulator test_sim;
        test_sim.reset(300, 66666);
        auto result = test_sim.step_batch({action});
        EXPECT_EQ(result.rewards.size(), 1);
    }
}
