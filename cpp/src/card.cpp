/**
 * @file card.cpp
 * @brief Implementation of card utilities and deck management.
 *
 * Provides:
 *   - Rank/suit name conversion for debugging output
 *   - Deck shuffling using std::mt19937_64 for determinism
 *   - Card drawing with automatic reshuffle when exhausted
 *   - Hand dealing and card replacement operations
 *
 * Determinism: The Deck class uses explicit seeding to ensure identical
 * seeds produce identical shuffle sequences across platforms.
 */

#include "../include/balatro/card.hpp"
#include <algorithm>

namespace balatro {

const char* get_rank_name(int rank) {
    static const char* names[] = {
        "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"
    };
    return (rank >= 0 && rank < NUM_RANKS) ? names[rank] : "?";
}

const char* get_suit_name(int suit) {
    static const char* names[] = {"♣", "♦", "♥", "♠"};
    return (suit >= 0 && suit < NUM_SUITS) ? names[suit] : "?";
}

Deck::Deck() : draw_idx_(0) {
    deck_.reserve(DECK_SIZE);
    discard_pile_.reserve(DECK_SIZE);
}

void Deck::reset(uint64_t seed) {
    deck_.clear();
    discard_pile_.clear();
    draw_idx_ = 0;

    // Initialize with all 52 cards
    for (int i = 0; i < DECK_SIZE; ++i) {
        deck_.push_back(static_cast<Card>(i));
    }

    // Shuffle deterministically
    rng_.seed(seed);
    std::shuffle(deck_.begin(), deck_.end(), rng_);
}

Card Deck::draw() {
    // Check if need to reshuffle
    if (is_empty()) {
        if (!discard_pile_.empty()) {
            reshuffle();
        } else {
            return 255; // Error: no cards available
        }
    }

    return deck_[draw_idx_++];
}

void Deck::discard(Card card) {
    discard_pile_.push_back(card);
}

void Deck::reshuffle() {
    // Move discards back into deck
    deck_.insert(deck_.end(), discard_pile_.begin(), discard_pile_.end());
    discard_pile_.clear();

    // Shuffle the new cards (not the entire deck)
    std::shuffle(deck_.begin() + draw_idx_, deck_.end(), rng_);
}

void deal_hand(Deck& deck, Hand& hand) {
    for (int i = 0; i < HAND_SIZE; ++i) {
        hand[i] = deck.draw();
    }
}

void replace_cards(Deck& deck, Hand& hand, const std::vector<int>& indices) {
    for (int idx : indices) {
        if (idx >= 0 && idx < HAND_SIZE) {
            // Discard old card and draw new one
            deck.discard(hand[idx]);
            hand[idx] = deck.draw();
        }
    }
}

} // namespace balatro
