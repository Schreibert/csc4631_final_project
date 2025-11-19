#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <random>

namespace balatro {

// Card encoding: 0-51 for standard 52-card deck
// Card = rank * 4 + suit
// Rank: 0=2, 1=3, ..., 8=10, 9=J, 10=Q, 11=K, 12=A
// Suit: 0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades
using Card = uint8_t;

constexpr int NUM_RANKS = 13;
constexpr int NUM_SUITS = 4;
constexpr int DECK_SIZE = 52;
constexpr int HAND_SIZE = 8;

// Rank and suit extraction
inline int get_rank(Card card) { return card / NUM_SUITS; }
inline int get_suit(Card card) { return card % NUM_SUITS; }
inline Card make_card(int rank, int suit) { return rank * NUM_SUITS + suit; }

// Rank values for scoring (2=0, 3=1, ..., 10=8, J=9, Q=10, K=11, A=12)
// Per the ruleset, we need to add rank values to base chips
constexpr int get_rank_value(int rank) {
    // Ranks 0-12 map to 2-A
    // For scoring: 2-10 add their face value, J/Q/K add 10, A adds 11
    if (rank <= 8) return rank + 2;  // 2-10
    if (rank <= 11) return 10;       // J, Q, K all worth 10
    return 11;                       // A worth 11
}

// Rank name for debugging
const char* get_rank_name(int rank);
const char* get_suit_name(int suit);

// Deck management
class Deck {
public:
    Deck();

    // Initialize deck with all 52 cards and shuffle with seed
    void reset(uint64_t seed);

    // Draw a card (returns 255 if deck empty - caller must check)
    Card draw();

    // Add card to discard pile
    void discard(Card card);

    // Reshuffle discards back into deck
    void reshuffle();

    // Check if deck is empty
    bool is_empty() const { return draw_idx_ >= deck_.size(); }

    // Get remaining cards in deck
    int remaining() const { return static_cast<int>(deck_.size()) - draw_idx_; }

private:
    std::vector<Card> deck_;
    std::vector<Card> discard_pile_;
    int draw_idx_;
    std::mt19937_64 rng_;
};

// Hand representation (fixed size 8)
using Hand = std::array<Card, HAND_SIZE>;

// Deal initial hand from deck
void deal_hand(Deck& deck, Hand& hand);

// Replace specific cards in hand by drawing from deck
void replace_cards(Deck& deck, Hand& hand, const std::vector<int>& indices);

} // namespace balatro
