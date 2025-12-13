/**
 * @file simulator.cpp
 * @brief Implementation of top-level simulator for Python bindings.
 *
 * Provides the main interface used by pybind11 to expose the simulator
 * to Python code. Wraps BlindState with batch execution and detailed
 * error messages for debugging.
 *
 * Key operations:
 *   - reset(): Initialize new episode with deterministic seeding
 *   - step_batch(): Execute multiple actions efficiently
 *   - validate_action(): Provide human-readable error messages
 *
 * Batch Execution:
 *   Actions are processed sequentially. If the episode ends (win/loss)
 *   before all actions are processed, remaining actions receive 0 reward
 *   and are not executed.
 */

#include "../include/balatro/simulator.hpp"
#include <sstream>

namespace balatro {

Simulator::Simulator() {}

Observation Simulator::reset(int target_score, uint64_t seed) {
    state_.reset(target_score, seed);
    return state_.get_observation();
}

StepBatchResult Simulator::step_batch(const std::vector<Action>& actions) {
    StepBatchResult result;
    result.rewards.resize(actions.size(), 0);
    result.done = false;
    result.win = false;

    for (size_t i = 0; i < actions.size(); ++i) {
        if (state_.is_done()) {
            // Already done - pad with zeros
            result.rewards[i] = 0;
            continue;
        }

        // Execute action
        int reward = execute_action(actions[i]);
        result.rewards[i] = reward;
    }

    result.final_obs = state_.get_observation();
    result.done = state_.is_done();
    result.win = state_.is_win();

    return result;
}

int Simulator::execute_action(const Action& action) {
    // Validate action first
    auto validation = validate_action(action);
    if (!validation.valid) {
        // Invalid action - return 0 reward
        return 0;
    }

    // Execute based on action type
    if (action.type == Action::PLAY) {
        return state_.play_hand(action);
    } else {
        state_.discard_cards(action);
        return 0; // Discards give no immediate reward
    }
}

ActionValidationResult Simulator::validate_action(const Action& action) const {
    // Count selected cards
    int count = 0;
    for (bool selected : action.card_mask) {
        if (selected) count++;
    }

    // Must select at least 1 card
    if (count == 0) {
        return ActionValidationResult(false, "Must select at least 1 card");
    }

    // If playing, must select at most 5 cards
    if (action.type == Action::PLAY && count > 5) {
        std::ostringstream oss;
        oss << "Cannot play more than 5 cards (selected " << count << ")";
        return ActionValidationResult(false, oss.str());
    }

    // If discarding, must select at most 5 cards
    if (action.type == Action::DISCARD && count > 5) {
        std::ostringstream oss;
        oss << "Cannot discard more than 5 cards (selected " << count << ")";
        return ActionValidationResult(false, oss.str());
    }

    // If playing, must have plays left
    if (action.type == Action::PLAY && state_.get_plays_left() <= 0) {
        return ActionValidationResult(false, "No plays remaining");
    }

    // If discarding, must have discards left
    if (action.type == Action::DISCARD && state_.get_discards_left() <= 0) {
        return ActionValidationResult(false, "No discards remaining");
    }

    // Can't take action if episode is done
    if (state_.is_done()) {
        return ActionValidationResult(false, "Episode is already finished");
    }

    return ActionValidationResult(true, "");
}

} // namespace balatro
