/**
 * @file scoring.hpp
 * @brief Balatro poker scoring formula implementation.
 *
 * Implements the Balatro scoring system where each poker hand earns chips
 * based on both hand type and the ranks of cards played.
 *
 * Scoring Formula:
 * @code
 *   score = (base_chips + rank_sum) × base_mult
 * @endcode
 *
 * Where:
 *   - base_chips: Fixed value per hand type (5 for High Card, 100 for Straight Flush)
 *   - rank_sum: Sum of get_rank_value() for scoring cards (2-10 face value, J/Q/K=10, A=11)
 *   - base_mult: Multiplier per hand type (1 for High Card, 8 for Straight Flush)
 *
 * Example: Full House K♣K♦K♥A♠A♣
 *   - base_chips = 40
 *   - rank_sum = 10 + 10 + 10 + 11 + 11 = 52
 *   - base_mult = 4
 *   - Total: (40 + 52) × 4 = 368 chips
 *
 * Base Values Table (Level 1):
 *   | Hand Type        | Base Chips | Multiplier |
 *   |------------------|------------|------------|
 *   | High Card        | 5          | 1          |
 *   | Pair             | 10         | 2          |
 *   | Two Pair         | 20         | 2          |
 *   | Three of a Kind  | 30         | 3          |
 *   | Straight         | 30         | 4          |
 *   | Flush            | 35         | 4          |
 *   | Full House       | 40         | 4          |
 *   | Four of a Kind   | 60         | 7          |
 *   | Straight Flush   | 100        | 8          |
 */

#pragma once

#include "hand_eval.hpp"

namespace balatro {

/**
 * @brief Base scoring values for a poker hand type.
 *
 * Each hand type has fixed base_chips and base_mult values that
 * determine scoring. Higher hand types have better base values.
 */
struct HandScoring {
    int base_chips; ///< Chips added before multiplication
    int base_mult;  ///< Multiplier applied to (base_chips + rank_sum)
};

/**
 * @brief Get base scoring values for a hand type.
 * @param type HandType enum value (HIGH_CARD through STRAIGHT_FLUSH)
 * @return HandScoring struct with base_chips and base_mult
 *
 * Returns Level 1 base values. See file header for complete table.
 */
HandScoring get_hand_scoring(HandType type);

/**
 * @brief Calculate total chip score for a played hand.
 * @param hand_eval Evaluated hand with type and rank_sum from evaluate_hand()
 * @return Total chips scored: (base_chips + rank_sum) × base_mult
 *
 * The rank_sum in HandEvaluation must use get_rank_value() scoring values
 * (not raw rank indices). This is handled by evaluate_hand() automatically.
 */
int calculate_score(const HandEvaluation& hand_eval);

} // namespace balatro
