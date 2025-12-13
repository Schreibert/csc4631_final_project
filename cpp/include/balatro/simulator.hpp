/**
 * @file simulator.hpp
 * @brief Top-level simulator interface for Python bindings.
 *
 * Provides the main entry point for the Balatro poker simulator,
 * designed for efficient integration with Python via pybind11.
 *
 * Key Features:
 *   - Batch action execution (reduces Python/C++ boundary crossings)
 *   - Detailed action validation with error messages
 *   - RL helper methods for hand analysis and action enumeration
 *
 * Usage from Python (via pybind11):
 * @code
 *   sim = _balatro_core.Simulator()
 *   obs = sim.reset(300, 42)  # target_score=300, seed=42
 *
 *   action = _balatro_core.Action()
 *   action.type = _balatro_core.ActionType.PLAY
 *   action.card_mask = [True, True, True, True, True, False, False, False]
 *
 *   result = sim.step_batch([action])
 *   print(f"Done: {result.done}, Win: {result.win}")
 * @endcode
 */

#pragma once

#include "blind_state.hpp"
#include <vector>
#include <cstdint>
#include <string>

namespace balatro {

/**
 * @brief Result of executing a batch of actions.
 *
 * Returned by step_batch() containing final state and per-action rewards.
 */
struct StepBatchResult {
    Observation final_obs;    ///< Observation after all actions executed
    std::vector<int> rewards; ///< Per-action chip deltas (raw, pre-shaping)
    bool done;                ///< True if episode ended during batch
    bool win;                 ///< True if episode was won (only valid when done)
};

/**
 * @brief Result of action validation with human-readable error.
 *
 * Returned by validate_action() for debugging invalid actions.
 */
struct ActionValidationResult {
    bool valid;              ///< True if action is legal
    std::string error_message; ///< Error description if invalid, empty if valid

    /** @brief Default constructor: valid action */
    ActionValidationResult() : valid(true), error_message("") {}

    /**
     * @brief Construct validation result.
     * @param v Whether action is valid
     * @param msg Error message (should be empty if v is true)
     */
    ActionValidationResult(bool v, const std::string& msg) : valid(v), error_message(msg) {}
};

/**
 * @brief Main simulator class exposed to Python.
 *
 * Wraps BlindState with a Python-friendly interface. Each Simulator
 * instance manages one episode at a time. Call reset() to start a new episode.
 *
 * Thread Safety: Not thread-safe. Create separate instances for parallel envs.
 */
class Simulator {
public:
    /** @brief Construct simulator (call reset() before use) */
    Simulator();

    /**
     * @brief Reset to start a new episode.
     * @param target_score Chips needed to win (e.g., 300)
     * @param seed Random seed for deterministic shuffling
     * @return Initial observation for the new episode
     *
     * Same (target_score, seed) always produces identical episode.
     */
    Observation reset(int target_score, uint64_t seed);

    /**
     * @brief Execute a batch of actions sequentially.
     * @param actions Vector of Action structs to execute in order
     * @return StepBatchResult with final observation and per-action rewards
     *
     * Actions are executed in sequence. If episode ends (win or loss),
     * remaining actions are skipped. Rewards vector contains chip delta
     * for each successfully executed action.
     *
     * Batching reduces Python/C++ call overhead for multi-step episodes.
     */
    StepBatchResult step_batch(const std::vector<Action>& actions);

    /**
     * @brief Validate an action with detailed error message.
     * @param action Action to validate
     * @return ActionValidationResult with valid flag and error message
     *
     * Useful for debugging: returns specific error like
     * "Cannot play: 0 plays remaining" or "Invalid card count: 6 (max 5)".
     */
    ActionValidationResult validate_action(const Action& action) const;

    /**
     * @brief Get read-only view of internal state.
     * @return Const reference to BlindState (for debugging/testing)
     */
    const BlindState& state_view() const { return state_; }

    // =========================================================================
    // RL Helper Methods (delegated to BlindState)
    // =========================================================================

    /**
     * @brief Get best possible 5-card hand from current state.
     * @return HandEvaluation with type, cards, and rank_sum
     * @see BlindState::get_best_hand()
     */
    HandEvaluation get_best_hand() const { return state_.get_best_hand(); }

    /**
     * @brief Predict chip score for a PLAY action without executing.
     * @param card_mask Boolean mask for card selection
     * @return Predicted chips, or 0 if invalid
     * @see BlindState::predict_play_score()
     */
    int predict_play_score(const std::array<bool, HAND_SIZE>& card_mask) const {
        return state_.predict_play_score(card_mask);
    }

    /**
     * @brief Enumerate all valid actions with predicted outcomes.
     * @return Vector of ActionOutcome sorted by predicted_chips
     * @see BlindState::enumerate_all_actions()
     */
    std::vector<ActionOutcome> enumerate_all_actions() const {
        return state_.enumerate_all_actions();
    }

private:
    /**
     * @brief Execute a single action and return reward.
     * @param action Action to execute
     * @return Chip delta for this action
     */
    int execute_action(const Action& action);

    BlindState state_; ///< Internal game state
};

} // namespace balatro
