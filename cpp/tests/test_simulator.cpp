#include <gtest/gtest.h>
#include "balatro/simulator.hpp"

using namespace balatro;

// Helper to create an action that selects all cards
Action make_play_all() {
    Action action;
    action.type = Action::PLAY;
    for (int i = 0; i < HAND_SIZE; ++i) {
        action.card_mask[i] = true;
    }
    return action;
}

// Helper to create an action that selects specific cards by indices
Action make_play(const std::vector<int>& indices) {
    Action action;
    action.type = Action::PLAY;
    for (int idx : indices) {
        if (idx >= 0 && idx < HAND_SIZE) {
            action.card_mask[idx] = true;
        }
    }
    return action;
}

// Helper to create a discard action
Action make_discard(const std::vector<int>& indices) {
    Action action;
    action.type = Action::DISCARD;
    for (int idx : indices) {
        if (idx >= 0 && idx < HAND_SIZE) {
            action.card_mask[idx] = true;
        }
    }
    return action;
}

TEST(SimulatorTest, Reset) {
    Simulator sim;
    auto obs = sim.reset(300, 12345);

    // Check initial state
    EXPECT_EQ(obs.plays_left, 4);
    EXPECT_EQ(obs.discards_left, 3);
    EXPECT_EQ(obs.chips_to_target, 300);
    EXPECT_EQ(obs.chips, 0);

    EXPECT_FALSE(sim.state_view().is_done());
    EXPECT_FALSE(sim.state_view().is_win());
}

TEST(SimulatorTest, DeterministicReset) {
    Simulator sim1, sim2;

    auto obs1 = sim1.reset(300, 99999);
    auto obs2 = sim2.reset(300, 99999);

    // Same seed should produce same initial observation
    EXPECT_EQ(obs1.plays_left, obs2.plays_left);
    EXPECT_EQ(obs1.discards_left, obs2.discards_left);
    EXPECT_EQ(obs1.chips, obs2.chips);
    EXPECT_EQ(obs1.chips_to_target, obs2.chips_to_target);

    // Same initial hand
    const auto& hand1 = sim1.state_view().get_hand();
    const auto& hand2 = sim2.state_view().get_hand();
    for (int i = 0; i < HAND_SIZE; ++i) {
        EXPECT_EQ(hand1[i], hand2[i]);
        EXPECT_EQ(obs1.card_ranks[i], obs2.card_ranks[i]);
        EXPECT_EQ(obs1.card_suits[i], obs2.card_suits[i]);
    }
}

TEST(SimulatorTest, StepBatch) {
    Simulator sim;
    sim.reset(5000, 54321); // High target so we don't win immediately

    // Execute some actions
    std::vector<Action> actions = {
        make_play({0, 1, 2, 3, 4}),  // Play first 5 cards
        make_discard({5, 6, 7})       // Discard last 3 cards
    };
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

        auto result = sim.step_batch({make_play({0, 1, 2, 3, 4})});
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
    std::vector<Action> actions = {
        make_play({0, 1, 2, 3, 4}),
        make_play({0, 1, 2, 3, 4}),
        make_play({0, 1, 2, 3, 4}),
        make_play({0, 1, 2, 3, 4})
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

    std::vector<Action> actions = {
        make_play({0, 1, 2, 3, 4}),
        make_discard({5, 6, 7}),
        make_play({0, 1}),
        make_play({0, 1, 2, 3, 4})
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

    EXPECT_EQ(result1.final_obs.chips, result2.final_obs.chips);
    EXPECT_EQ(result1.final_obs.plays_left, result2.final_obs.plays_left);
    EXPECT_EQ(result1.final_obs.discards_left, result2.final_obs.discards_left);
}

TEST(SimulatorTest, PaddingAfterDone) {
    Simulator sim;
    sim.reset(10, 33333); // Low target

    // Win quickly
    auto result1 = sim.step_batch({make_play({0, 1, 2, 3, 4})});

    if (!result1.done) {
        // If not done yet, play once more
        result1 = sim.step_batch({make_play({0, 1, 2, 3, 4})});
    }

    ASSERT_TRUE(sim.state_view().is_done());

    // Now execute more actions - should be padded with 0 rewards
    auto result2 = sim.step_batch({make_play({0, 1, 2, 3, 4}), make_play({0, 1, 2, 3, 4})});

    EXPECT_EQ(result2.rewards.size(), 2);
    EXPECT_EQ(result2.rewards[0], 0);
    EXPECT_EQ(result2.rewards[1], 0);
}

TEST(SimulatorTest, ObservationValidity) {
    Simulator sim;
    auto obs = sim.reset(500, 44444);

    // Check observation ranges
    EXPECT_GE(obs.plays_left, 0);
    EXPECT_LE(obs.plays_left, 4);
    EXPECT_GE(obs.discards_left, 0);
    EXPECT_LE(obs.discards_left, 3);
    EXPECT_GE(obs.chips_to_target, 0);

    // Check hand observation
    for (int i = 0; i < HAND_SIZE; ++i) {
        EXPECT_GE(obs.card_ranks[i], 0);
        EXPECT_LE(obs.card_ranks[i], 12); // 0-12 for 2-A
        EXPECT_GE(obs.card_suits[i], 0);
        EXPECT_LE(obs.card_suits[i], 3);  // 0-3 for 4 suits
    }
}

TEST(SimulatorTest, PlayAndDiscard) {
    Simulator sim;
    sim.reset(300, 55555);

    // Test playing 5 cards
    auto result1 = sim.step_batch({make_play({0, 1, 2, 3, 4})});
    EXPECT_EQ(result1.rewards.size(), 1);
    EXPECT_GE(result1.rewards[0], 0); // Should give some reward
    EXPECT_EQ(sim.state_view().get_plays_left(), 3);

    // Test discarding 3 cards
    auto result2 = sim.step_batch({make_discard({0, 1, 2})});
    EXPECT_EQ(result2.rewards.size(), 1);
    EXPECT_EQ(result2.rewards[0], 0); // Discard gives no reward
    EXPECT_EQ(sim.state_view().get_discards_left(), 2);
}

TEST(SimulatorTest, InvalidActionValidation) {
    Simulator sim;
    sim.reset(300, 66666);

    // Try to play 0 cards (invalid)
    Action empty_play;
    empty_play.type = Action::PLAY;
    // card_mask is all false

    auto validation = sim.validate_action(empty_play);
    EXPECT_FALSE(validation.valid);

    // Try to play too many cards (invalid - poker hands are max 5 cards)
    Action too_many = make_play({0, 1, 2, 3, 4, 5, 6, 7});
    validation = sim.validate_action(too_many);
    EXPECT_FALSE(validation.valid);
}
