#include "../include/balatro/simulator.hpp"
#include "../include/balatro/hand_eval.hpp"
#include "../include/balatro/scoring.hpp"
#include <cassert>
#include <iostream>
#include <vector>

using namespace balatro;

/**
 * Golden tests for enhanced observation features.
 *
 * Tests:
 * 1. best_hand_type detection for known hands
 * 2. best_hand_score calculation
 * 3. Hand pattern flags (has_full_house, etc.)
 * 4. Score prediction accuracy
 * 5. Action enumeration count
 */

void test_best_hand_detection() {
    std::cout << "Test 1: Best hand type detection..." << std::endl;

    Simulator sim;

    // Test with known seed that produces a specific hand
    // Seed 42 should give us a specific starting hand
    Observation obs = sim.reset(300, 42);

    // Verify best_hand_type is set
    assert(obs.best_hand_type >= 0 && obs.best_hand_type <= 8);
    assert(obs.best_hand_score >= 0);

    std::cout << "  best_hand_type: " << obs.best_hand_type
              << " (" << hand_type_name(static_cast<HandType>(obs.best_hand_type)) << ")" << std::endl;
    std::cout << "  best_hand_score: " << obs.best_hand_score << std::endl;

    // Verify get_best_hand() returns consistent result
    HandEvaluation best = sim.get_best_hand();
    assert(static_cast<int>(best.type) == obs.best_hand_type);

    std::cout << "  [PASS]" << std::endl;
}

void test_hand_pattern_flags() {
    std::cout << "Test 2: Hand pattern flags consistency..." << std::endl;

    Simulator sim;
    Observation obs = sim.reset(300, 42);

    // Verify boolean flags are consistent with best_hand_type
    HandType best_type = static_cast<HandType>(obs.best_hand_type);

    // If has_straight_flush, must also have has_straight and has_flush
    if (obs.has_straight_flush) {
        assert(obs.has_straight);
        assert(obs.has_flush);
        assert(best_type == HandType::STRAIGHT_FLUSH);
    }

    // If has_full_house, best hand must be at least full house
    if (obs.has_full_house) {
        assert(best_type == HandType::FULL_HOUSE || best_type == HandType::STRAIGHT_FLUSH);
    }

    // If has_four_of_kind, best hand must be at least four of a kind
    if (obs.has_four_of_kind) {
        assert(best_type == HandType::FOUR_OF_A_KIND || best_type == HandType::STRAIGHT_FLUSH);
    }

    std::cout << "  has_pair: " << obs.has_pair << std::endl;
    std::cout << "  has_trips: " << obs.has_trips << std::endl;
    std::cout << "  has_two_pair: " << obs.has_two_pair << std::endl;
    std::cout << "  has_full_house: " << obs.has_full_house << std::endl;
    std::cout << "  has_four_of_kind: " << obs.has_four_of_kind << std::endl;
    std::cout << "  has_straight: " << obs.has_straight << std::endl;
    std::cout << "  has_flush: " << obs.has_flush << std::endl;
    std::cout << "  has_straight_flush: " << obs.has_straight_flush << std::endl;

    std::cout << "  [PASS]" << std::endl;
}

void test_score_prediction() {
    std::cout << "Test 3: Score prediction accuracy..." << std::endl;

    Simulator sim;
    sim.reset(300, 100);  // Use seed 100 for variety

    // Create a PLAY action (play first 5 cards)
    Action action;
    action.type = Action::PLAY;
    action.card_mask = {true, true, true, true, true, false, false, false};

    // Predict score
    int predicted = sim.predict_play_score(action.card_mask);
    assert(predicted > 0);  // Should score something

    // Execute action and compare
    StepBatchResult result = sim.step_batch({action});
    int actual = result.rewards[0];

    std::cout << "  Predicted score: " << predicted << std::endl;
    std::cout << "  Actual score: " << actual << std::endl;

    // Must match exactly!
    assert(predicted == actual);

    std::cout << "  [PASS] Prediction matches execution" << std::endl;
}

void test_action_enumeration() {
    std::cout << "Test 4: Action enumeration..." << std::endl;

    Simulator sim;
    sim.reset(300, 200);  // Fresh game

    // Enumerate all actions
    std::vector<ActionOutcome> outcomes = sim.enumerate_all_actions();

    assert(!outcomes.empty());
    std::cout << "  Total actions enumerated: " << outcomes.size() << std::endl;

    // Count PLAY vs DISCARD
    int play_count = 0;
    int discard_count = 0;
    for (const auto& outcome : outcomes) {
        assert(outcome.valid);  // All should be valid
        if (outcome.action.type == Action::PLAY) {
            play_count++;
            assert(outcome.predicted_chips > 0);  // PLAY should score chips
        } else {
            discard_count++;
            assert(outcome.predicted_chips == 0);  // DISCARD scores nothing
        }
    }

    std::cout << "  PLAY actions: " << play_count << std::endl;
    std::cout << "  DISCARD actions: " << discard_count << std::endl;

    // At start: 4 plays left, 3 discards left
    // PLAY: C(8,1) + C(8,2) + C(8,3) + C(8,4) + C(8,5) = 8 + 28 + 56 + 70 + 56 = 218
    // DISCARD: C(8,1) + C(8,2) + ... + C(8,8) = 2^8 - 1 = 255
    // Total: 218 + 255 = 473
    assert(play_count == 218);
    assert(discard_count == 255);
    assert(outcomes.size() == 473);

    // Verify sorted by score (descending)
    for (size_t i = 0; i < outcomes.size() - 1; ++i) {
        if (outcomes[i].action.type == Action::PLAY &&
            outcomes[i+1].action.type == Action::PLAY) {
            assert(outcomes[i].predicted_chips >= outcomes[i+1].predicted_chips);
        }
    }

    std::cout << "  [PASS] Correct count and sorting" << std::endl;
}

void test_best_hand_consistency() {
    std::cout << "Test 5: Best hand score consistency..." << std::endl;

    Simulator sim;
    Observation obs = sim.reset(300, 300);

    // Get best hand score from observation
    int obs_score = obs.best_hand_score;

    // Get best hand via method
    HandEvaluation best = sim.get_best_hand();
    int method_score = calculate_score(best);

    std::cout << "  Observation score: " << obs_score << std::endl;
    std::cout << "  Method score: " << method_score << std::endl;

    // Must match!
    assert(obs_score == method_score);

    std::cout << "  [PASS] Scores are consistent" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Enhanced Observation Golden Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    try {
        test_best_hand_detection();
        test_hand_pattern_flags();
        test_score_prediction();
        test_action_enumeration();
        test_best_hand_consistency();

        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "ALL TESTS PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "TEST FAILED: Unknown error" << std::endl;
        return 1;
    }
}
