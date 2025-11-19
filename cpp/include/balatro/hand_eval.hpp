#pragma once

#include "balatro/card.hpp"
#include <vector>
#include <array>

namespace balatro {

// Poker hand types (ordered by strength)
enum class HandType {
    HIGH_CARD = 0,
    PAIR = 1,
    TWO_PAIR = 2,
    THREE_OF_A_KIND = 3,
    STRAIGHT = 4,
    FLUSH = 5,
    FULL_HOUSE = 6,
    FOUR_OF_A_KIND = 7,
    STRAIGHT_FLUSH = 8
};

const char* hand_type_name(HandType type);

// Result of hand evaluation
struct HandEvaluation {
    HandType type;
    std::vector<Card> cards_used; // The 1-5 cards that form the hand
    int rank_sum;                 // Sum of rank values for scoring

    HandEvaluation() : type(HandType::HIGH_CARD), rank_sum(0) {}
};

// Evaluate a poker hand from a subset of cards
// cards must be between 1-5 cards
HandEvaluation evaluate_hand(const std::vector<Card>& cards);

// Find the best possible poker hand from a collection (e.g., 8-card hand)
// Returns the best 5-card (or fewer) combination
HandEvaluation find_best_hand(const std::vector<Card>& cards);

// Helper functions for observation features

// Check if a hand contains at least one pair
bool has_pair(const std::array<Card, HAND_SIZE>& hand);

// Check if a hand contains at least three of a kind
bool has_three_of_kind(const std::array<Card, HAND_SIZE>& hand);

// Check if hand has straight potential (one card away from straight)
bool has_straight_potential(const std::array<Card, HAND_SIZE>& hand);

// Check if hand has flush potential (4+ cards of same suit)
bool has_flush_potential(const std::array<Card, HAND_SIZE>& hand);

// Get the max rank in hand, bucketed (0-5)
// 0: 2-4, 1: 5-7, 2: 8-10, 3: J-Q, 4: K, 5: A
int get_max_rank_bucket(const std::array<Card, HAND_SIZE>& hand);

} // namespace balatro
