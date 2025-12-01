#include "../include/balatro/blind_state.hpp"
#include <iostream>

int main() {
    using namespace balatro;

    BlindState state;
    state.reset(300, 42);

    Observation obs = state.get_observation();

    std::cout << "Card arrays from observation:\n";
    for (int i = 0; i < HAND_SIZE; ++i) {
        std::cout << "  Card " << i << ": rank=" << obs.card_ranks[i]
                  << " suit=" << obs.card_suits[i] << "\n";
    }

    std::cout << "\nDirect hand access:\n";
    const Hand& hand = state.get_hand();
    for (int i = 0; i < HAND_SIZE; ++i) {
        std::cout << "  Card " << i << ": rank=" << get_rank(hand[i])
                  << " suit=" << get_suit(hand[i]) << "\n";
    }

    return 0;
}
