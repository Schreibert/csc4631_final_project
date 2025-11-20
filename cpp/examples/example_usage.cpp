// Example usage of the new direct card control interface

#include "../include/balatro/simulator.hpp"
#include "../include/balatro/card.hpp"
#include <iostream>
#include <iomanip>

using namespace balatro;

void print_observation(const Observation& obs) {
    std::cout << "\n=== Observation ===" << std::endl;
    std::cout << "Plays left: " << obs.plays_left << std::endl;
    std::cout << "Discards left: " << obs.discards_left << std::endl;
    std::cout << "Current chips: " << obs.chips << std::endl;
    std::cout << "Chips to target: " << obs.chips_to_target << std::endl;
    std::cout << "Deck remaining: " << obs.deck_remaining << std::endl;
    std::cout << "\nHand analysis:" << std::endl;
    std::cout << "  Has pair: " << obs.has_pair << std::endl;
    std::cout << "  Has trips: " << obs.has_trips << std::endl;
    std::cout << "  Straight potential: " << obs.straight_potential << std::endl;
    std::cout << "  Flush potential: " << obs.flush_potential << std::endl;

    std::cout << "\nCurrent hand:" << std::endl;
    const char* rank_names[] = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"};
    const char* suit_symbols[] = {"♣", "♦", "♥", "♠"};

    for (int i = 0; i < HAND_SIZE; ++i) {
        std::cout << "  [" << i << "] " << std::setw(2) << rank_names[obs.card_ranks[i]]
                  << suit_symbols[obs.card_suits[i]] << std::endl;
    }
}

int main() {
    std::cout << "=== Balatro Simulator - Direct Card Control Example ===" << std::endl;

    // Create simulator
    Simulator sim;

    // Reset with target score of 300 and seed 42
    std::cout << "\nResetting game with target_score=300, seed=42" << std::endl;
    Observation obs = sim.reset(300, 42);
    print_observation(obs);

    // Example 1: Play the first 5 cards as a poker hand
    std::cout << "\n\n--- Example 1: Playing first 5 cards ---" << std::endl;
    Action play_action;
    play_action.type = Action::PLAY;
    play_action.card_mask = {true, true, true, true, true, false, false, false};

    // Validate before executing
    auto validation = sim.validate_action(play_action);
    if (validation.valid) {
        std::cout << "Action is valid! Executing..." << std::endl;
        auto result = sim.step_batch({play_action});
        std::cout << "Reward: " << result.rewards[0] << std::endl;
        std::cout << "Done: " << result.done << ", Win: " << result.win << std::endl;
        print_observation(result.final_obs);
    } else {
        std::cout << "Invalid action: " << validation.error_message << std::endl;
    }

    // Example 2: Discard 3 lowest cards (indices 5, 6, 7)
    std::cout << "\n\n--- Example 2: Discarding 3 cards ---" << std::endl;
    Action discard_action;
    discard_action.type = Action::DISCARD;
    discard_action.card_mask = {false, false, false, false, false, true, true, true};

    validation = sim.validate_action(discard_action);
    if (validation.valid) {
        std::cout << "Action is valid! Executing..." << std::endl;
        auto result = sim.step_batch({discard_action});
        std::cout << "Reward: " << result.rewards[0] << " (discards give no reward)" << std::endl;
        print_observation(result.final_obs);
    } else {
        std::cout << "Invalid action: " << validation.error_message << std::endl;
    }

    // Example 3: Invalid action - playing 0 cards
    std::cout << "\n\n--- Example 3: Invalid action (0 cards) ---" << std::endl;
    Action invalid_action;
    invalid_action.type = Action::PLAY;
    invalid_action.card_mask = {false, false, false, false, false, false, false, false};

    validation = sim.validate_action(invalid_action);
    if (!validation.valid) {
        std::cout << "Correctly detected invalid action: " << validation.error_message << std::endl;
    }

    // Example 4: Invalid action - playing 6 cards
    std::cout << "\n\n--- Example 4: Invalid action (6 cards) ---" << std::endl;
    Action invalid_action2;
    invalid_action2.type = Action::PLAY;
    invalid_action2.card_mask = {true, true, true, true, true, true, false, false};

    validation = sim.validate_action(invalid_action2);
    if (!validation.valid) {
        std::cout << "Correctly detected invalid action: " << validation.error_message << std::endl;
    }

    // Example 5: Batch execution
    std::cout << "\n\n--- Example 5: Batch execution (play + discard + play) ---" << std::endl;

    // Reset for clean batch test
    obs = sim.reset(500, 123);
    std::cout << "Reset with target_score=500, seed=123" << std::endl;

    std::vector<Action> batch_actions;

    // Action 1: Play 3 cards
    Action a1;
    a1.type = Action::PLAY;
    a1.card_mask = {true, false, true, false, true, false, false, false};
    batch_actions.push_back(a1);

    // Action 2: Discard 2 cards
    Action a2;
    a2.type = Action::DISCARD;
    a2.card_mask = {false, true, false, false, false, true, false, false};
    batch_actions.push_back(a2);

    // Action 3: Play 5 cards
    Action a3;
    a3.type = Action::PLAY;
    a3.card_mask = {true, true, true, true, true, false, false, false};
    batch_actions.push_back(a3);

    auto result = sim.step_batch(batch_actions);
    std::cout << "Batch executed!" << std::endl;
    std::cout << "Rewards: [";
    for (size_t i = 0; i < result.rewards.size(); ++i) {
        std::cout << result.rewards[i];
        if (i < result.rewards.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Final state - Done: " << result.done << ", Win: " << result.win << std::endl;
    print_observation(result.final_obs);

    std::cout << "\n=== Example Complete ===" << std::endl;
    return 0;
}
