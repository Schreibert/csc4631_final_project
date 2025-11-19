#pragma once

#include "balatro/blind_state.hpp"
#include <vector>
#include <cstdint>

namespace balatro {

// Action codes (v0 contract)
constexpr int ACTION_PLAY_BEST = 0;
constexpr int ACTION_PLAY_PAIR = 1;
constexpr int ACTION_DISCARD_NON_PAIRED = 2;
constexpr int ACTION_DISCARD_LOWEST_3 = 3;
constexpr int ACTION_PLAY_HIGHEST_3 = 4;
constexpr int ACTION_RANDOM_PLAY = 5;
constexpr int ACTION_RANDOM_DISCARD = 6;
constexpr int NUM_ACTIONS = 7;

// Result of step_batch
struct StepBatchResult {
    Observation final_obs;
    std::vector<int> rewards;
    bool done;
    bool win;
};

// Main simulator class
class Simulator {
public:
    Simulator();

    // Reset to new episode
    Observation reset(int target_score, uint64_t seed);

    // Execute batch of actions
    StepBatchResult step_batch(const std::vector<int>& actions);

    // Get read-only state view (for debugging/testing)
    const BlindState& state_view() const { return state_; }

private:
    // Execute single action
    int execute_action(int action_code);

    // Action implementations
    std::vector<int> action_play_best();
    std::vector<int> action_play_pair();
    std::vector<int> action_discard_non_paired();
    std::vector<int> action_discard_lowest_3();
    std::vector<int> action_play_highest_3();
    std::vector<int> action_random_play();
    std::vector<int> action_random_discard();

    BlindState state_;
    std::mt19937_64 action_rng_; // For random actions
};

} // namespace balatro
