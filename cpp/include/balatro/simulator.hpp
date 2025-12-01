#pragma once

#include "blind_state.hpp"
#include <vector>
#include <cstdint>
#include <string>

namespace balatro {

// Result of step_batch
struct StepBatchResult {
    Observation final_obs;
    std::vector<int> rewards;
    bool done;
    bool win;
};

// Action validation result
struct ActionValidationResult {
    bool valid;
    std::string error_message;

    ActionValidationResult() : valid(true), error_message("") {}
    ActionValidationResult(bool v, const std::string& msg) : valid(v), error_message(msg) {}
};

// Main simulator class
class Simulator {
public:
    Simulator();

    // Reset to new episode
    Observation reset(int target_score, uint64_t seed);

    // Execute batch of actions with direct card selection
    StepBatchResult step_batch(const std::vector<Action>& actions);

    // Validate an action (returns detailed error if invalid)
    ActionValidationResult validate_action(const Action& action) const;

    // Get read-only state view (for debugging/testing)
    const BlindState& state_view() const { return state_; }

    // RL helper methods (delegate to BlindState)

    // Get best possible hand from current state
    HandEvaluation get_best_hand() const { return state_.get_best_hand(); }

    // Predict score for a PLAY action
    int predict_play_score(const std::array<bool, HAND_SIZE>& card_mask) const {
        return state_.predict_play_score(card_mask);
    }

    // Enumerate all valid actions with predicted outcomes
    std::vector<ActionOutcome> enumerate_all_actions() const {
        return state_.enumerate_all_actions();
    }

private:
    // Execute single action (returns reward)
    int execute_action(const Action& action);

    BlindState state_;
};

} // namespace balatro
