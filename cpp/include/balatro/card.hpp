/**
 * @file card.hpp
 * @brief Card encoding, deck management, and hand representation for Balatro poker.
 *
 * This file defines the fundamental card data structures used throughout the simulator.
 * Cards are encoded as single bytes (0-51) using the formula: rank * 4 + suit.
 * This compact encoding enables efficient storage and fast rank/suit extraction.
 *
 * Encoding scheme:
 *   - Ranks: 0=2, 1=3, 2=4, ..., 8=10, 9=J, 10=Q, 11=K, 12=A
 *   - Suits: 0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades
 *   - Example: Ace of Spades = 12 * 4 + 3 = 51
 *   - Example: Two of Clubs = 0 * 4 + 0 = 0
 *
 * The Deck class provides deterministic shuffling using std::mt19937_64,
 * ensuring identical seeds produce identical card orderings for reproducibility.
 */

#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <random>

namespace balatro {

/**
 * @brief Card type encoded as a single byte (0-51).
 *
 * Card value = rank * 4 + suit, where:
 *   - rank: 0-12 (2 through Ace)
 *   - suit: 0-3 (Clubs, Diamonds, Hearts, Spades)
 *
 * Use get_rank() and get_suit() to extract components.
 * Use make_card() to construct from rank and suit.
 */
using Card = uint8_t;

/** @brief Total number of ranks in a standard deck (2 through Ace) */
constexpr int NUM_RANKS = 13;

/** @brief Total number of suits (Clubs, Diamonds, Hearts, Spades) */
constexpr int NUM_SUITS = 4;

/** @brief Total cards in a standard deck */
constexpr int DECK_SIZE = 52;

/** @brief Fixed hand size for Balatro v0 ruleset */
constexpr int HAND_SIZE = 8;

/**
 * @brief Extract rank from card encoding.
 * @param card Card value (0-51)
 * @return Rank index (0-12, where 0=2, 12=Ace)
 */
inline int get_rank(Card card) { return card / NUM_SUITS; }

/**
 * @brief Extract suit from card encoding.
 * @param card Card value (0-51)
 * @return Suit index (0=Clubs, 1=Diamonds, 2=Hearts, 3=Spades)
 */
inline int get_suit(Card card) { return card % NUM_SUITS; }

/**
 * @brief Construct a card from rank and suit indices.
 * @param rank Rank index (0-12)
 * @param suit Suit index (0-3)
 * @return Encoded card value (0-51)
 */
inline Card make_card(int rank, int suit) {
    return static_cast<Card>(rank * NUM_SUITS + suit);
}

/**
 * @brief Get the scoring value for a rank (used in Balatro scoring formula).
 *
 * Scoring values differ from rank indices:
 *   - Ranks 0-8 (cards 2-10): Face value (2, 3, 4, ..., 10)
 *   - Ranks 9-11 (J, Q, K): Value 10
 *   - Rank 12 (Ace): Value 11
 *
 * @param rank Rank index (0-12)
 * @return Scoring value (2-11) for use in chip calculations
 */
constexpr int get_rank_value(int rank) {
    if (rank <= 8) return rank + 2;  // 2-10: face value
    if (rank <= 11) return 10;       // J, Q, K: all worth 10
    return 11;                       // Ace: worth 11
}

/**
 * @brief Get human-readable rank name for debugging.
 * @param rank Rank index (0-12)
 * @return Rank name string ("2", "3", ..., "J", "Q", "K", "A")
 */
const char* get_rank_name(int rank);

/**
 * @brief Get human-readable suit name for debugging.
 * @param suit Suit index (0-3)
 * @return Suit name string ("Clubs", "Diamonds", "Hearts", "Spades")
 */
const char* get_suit_name(int suit);

/**
 * @brief Manages a 52-card deck with deterministic shuffling and discard pile.
 *
 * The Deck class handles all card drawing, discarding, and reshuffling operations
 * with explicit seeding for determinism. Uses std::mt19937_64 to ensure identical
 * seeds produce identical card orderings across platforms.
 *
 * Key features:
 *   - Deterministic shuffle based on seed (critical for RL reproducibility)
 *   - Automatic reshuffle when deck is exhausted (discards return to deck)
 *   - Discard pile tracking for observation features
 *
 * Usage:
 * @code
 *   Deck deck;
 *   deck.reset(42);  // Seed 42 always produces same shuffle
 *   Card c1 = deck.draw();
 *   deck.discard(c1);
 *   deck.reshuffle();  // Discarded cards return to deck
 * @endcode
 */
class Deck {
public:
    /** @brief Construct an empty deck (call reset() before use) */
    Deck();

    /**
     * @brief Initialize deck with all 52 cards and shuffle deterministically.
     * @param seed Random seed for shuffle (same seed = same card order)
     *
     * Clears any existing state, creates fresh 52-card deck, and shuffles
     * using the provided seed. Discard pile is also cleared.
     */
    void reset(uint64_t seed);

    /**
     * @brief Draw one card from the top of the deck.
     * @return Card value (0-51), or 255 if deck and discard pile are both empty
     *
     * If deck is empty but discard pile has cards, automatically reshuffles
     * before drawing. Returns 255 only when no cards remain anywhere.
     */
    Card draw();

    /**
     * @brief Add a card to the discard pile.
     * @param card Card value to discard (0-51)
     *
     * Discarded cards remain out of play until reshuffle() is called
     * (either explicitly or automatically when deck empties).
     */
    void discard(Card card);

    /**
     * @brief Reshuffle discard pile back into the deck.
     *
     * Moves all cards from discard pile into deck and shuffles.
     * Uses the same RNG instance (continues from current state).
     */
    void reshuffle();

    /**
     * @brief Check if the draw pile is empty.
     * @return True if no cards remain in draw pile (discards may still exist)
     */
    bool is_empty() const { return draw_idx_ >= deck_.size(); }

    /**
     * @brief Get count of cards remaining in draw pile.
     * @return Number of cards that can be drawn before reshuffle needed
     */
    int remaining() const { return static_cast<int>(deck_.size()) - draw_idx_; }

private:
    std::vector<Card> deck_;         ///< Draw pile (cards available to draw)
    std::vector<Card> discard_pile_; ///< Discarded cards (out of play)
    int draw_idx_;                   ///< Index of next card to draw
    std::mt19937_64 rng_;            ///< Mersenne Twister RNG for shuffling
};

/**
 * @brief Fixed-size hand representation (8 cards for Balatro v0).
 *
 * Hand is always exactly HAND_SIZE cards. Cards are stored as encoded
 * values (0-51). Use get_rank() and get_suit() to extract components.
 */
using Hand = std::array<Card, HAND_SIZE>;

/**
 * @brief Deal a full hand of HAND_SIZE cards from the deck.
 * @param deck Source deck to draw from (modified)
 * @param hand Destination hand array (overwritten)
 *
 * Draws exactly HAND_SIZE cards from deck into hand array.
 * Deck must have sufficient cards (or discard pile for reshuffle).
 */
void deal_hand(Deck& deck, Hand& hand);

/**
 * @brief Replace specific cards in hand by drawing from deck.
 * @param deck Source deck for new cards (modified)
 * @param hand Hand to modify (cards at indices replaced)
 * @param indices Vector of hand positions (0-7) to replace
 *
 * For each index in indices, the card at that position is replaced
 * with a newly drawn card from the deck. Original cards are NOT
 * automatically discarded (caller must handle if needed).
 */
void replace_cards(Deck& deck, Hand& hand, const std::vector<int>& indices);

} // namespace balatro
