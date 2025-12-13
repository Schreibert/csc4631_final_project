/**
 * @file hand_eval.hpp
 * @brief Poker hand evaluation and pattern detection algorithms.
 *
 * Implements the standard 9-hand poker ranking system used in Balatro:
 *   1. HIGH_CARD (weakest)
 *   2. PAIR
 *   3. TWO_PAIR
 *   4. THREE_OF_A_KIND
 *   5. STRAIGHT
 *   6. FLUSH
 *   7. FULL_HOUSE
 *   8. FOUR_OF_A_KIND
 *   9. STRAIGHT_FLUSH (strongest)
 *
 * Key algorithms:
 *   - evaluate_hand(): Evaluates 1-5 cards for hand type
 *   - find_best_hand(): Finds best 5-card hand from 8 cards (C(8,5) = 56 combinations)
 *   - Pattern detection helpers for observation features (flush potential, straight draws)
 *
 * Special cases handled:
 *   - A-2-3-4-5 "wheel" straight (Ace acts as low card)
 *   - Straight Flush detection (straight AND flush combined)
 */

#pragma once

#include "card.hpp"
#include <vector>
#include <array>

namespace balatro {

/**
 * @brief Poker hand types ordered by strength (0 = weakest, 8 = strongest).
 *
 * Standard poker hand rankings used for both hand evaluation and scoring.
 * Higher enum values indicate stronger hands worth more chips.
 */
enum class HandType {
    HIGH_CARD = 0,        ///< No matching cards (single highest card)
    PAIR = 1,             ///< Two cards of same rank
    TWO_PAIR = 2,         ///< Two different pairs
    THREE_OF_A_KIND = 3,  ///< Three cards of same rank (trips)
    STRAIGHT = 4,         ///< Five consecutive ranks (includes A-2-3-4-5 wheel)
    FLUSH = 5,            ///< Five cards of same suit
    FULL_HOUSE = 6,       ///< Three of a kind plus a pair
    FOUR_OF_A_KIND = 7,   ///< Four cards of same rank (quads)
    STRAIGHT_FLUSH = 8    ///< Straight and flush combined (strongest)
};

/**
 * @brief Get human-readable name for a hand type.
 * @param type HandType enum value
 * @return String name ("High Card", "Pair", "Full House", etc.)
 */
const char* hand_type_name(HandType type);

/**
 * @brief Result of evaluating a poker hand.
 *
 * Contains the hand type, the cards that form the hand, and the sum
 * of rank values used in Balatro's scoring formula.
 */
struct HandEvaluation {
    HandType type;                ///< Detected hand type (0-8)
    std::vector<Card> cards_used; ///< The 1-5 cards forming this hand
    int rank_sum;                 ///< Sum of get_rank_value() for scoring cards

    /** @brief Default constructor (HIGH_CARD with no cards) */
    HandEvaluation() : type(HandType::HIGH_CARD), rank_sum(0) {}
};

/**
 * @brief Evaluate a poker hand from 1-5 cards.
 * @param cards Vector of 1-5 cards to evaluate
 * @return HandEvaluation with detected type, cards used, and rank sum
 *
 * Algorithm:
 *   1. Sort cards by rank
 *   2. Check for flush (all same suit, requires 5 cards)
 *   3. Check for straight (5 consecutive ranks, handles A-2-3-4-5 wheel)
 *   4. Count rank frequencies for pairs/trips/quads
 *   5. Return highest-ranking hand found
 *
 * @note With fewer than 5 cards, straights and flushes are impossible.
 *       The function still detects pairs, trips, etc.
 */
HandEvaluation evaluate_hand(const std::vector<Card>& cards);

/**
 * @brief Find the best possible 5-card poker hand from a larger set.
 * @param cards Vector of cards (typically 8 for Balatro hand)
 * @return Best HandEvaluation found among all valid subsets
 *
 * Uses combinatorial search to find the strongest hand:
 *   - First tries all 5-card combinations: C(8,5) = 56 combos
 *   - Then 4-card, 3-card, etc. if needed
 *   - Returns the combination with highest HandType
 *
 * Complexity: O(C(n,5)) where n = cards.size(), typically ~56 evaluations.
 */
HandEvaluation find_best_hand(const std::vector<Card>& cards);

// ============================================================================
// Observation Feature Helpers
// ============================================================================
// These functions compute features for the RL observation vector.
// They analyze the full 8-card hand without selecting specific subsets.

/**
 * @brief Check if hand contains at least one pair.
 * @param hand Fixed 8-card hand array
 * @return True if any two cards share the same rank
 */
bool has_pair(const std::array<Card, HAND_SIZE>& hand);

/**
 * @brief Check if hand contains three-of-a-kind or better.
 * @param hand Fixed 8-card hand array
 * @return True if any three cards share the same rank
 */
bool has_three_of_kind(const std::array<Card, HAND_SIZE>& hand);

/**
 * @brief Check if hand has straight potential (one card away from straight).
 * @param hand Fixed 8-card hand array
 * @return True if hand contains 4 cards within 5 consecutive ranks
 *
 * Detects "open-ended" and "gutshot" straight draws where one more
 * card would complete a straight. Used as observation feature for RL.
 */
bool has_straight_potential(const std::array<Card, HAND_SIZE>& hand);

/**
 * @brief Check if hand has flush potential (4+ cards of same suit).
 * @param hand Fixed 8-card hand array
 * @return True if 4 or more cards share the same suit
 *
 * A flush draw that could be completed with discards.
 * Used as observation feature for RL agents.
 */
bool has_flush_potential(const std::array<Card, HAND_SIZE>& hand);

/**
 * @brief Get the highest rank in hand, bucketed into 6 categories.
 * @param hand Fixed 8-card hand array
 * @return Bucket index (0-5)
 *
 * Bucket mapping:
 *   - 0: 2-4 (ranks 0-2)
 *   - 1: 5-7 (ranks 3-5)
 *   - 2: 8-10 (ranks 6-8)
 *   - 3: J-Q (ranks 9-10)
 *   - 4: K (rank 11)
 *   - 5: A (rank 12)
 *
 * Used to discretize high card info for tabular RL methods.
 */
int get_max_rank_bucket(const std::array<Card, HAND_SIZE>& hand);

} // namespace balatro
