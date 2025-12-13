/**
 * @file scoring.cpp
 * @brief Implementation of Balatro scoring formula.
 *
 * Scoring Formula: score = (base_chips + rank_sum) × base_mult
 *
 * The base values are from Balatro Level 1. In the full game, these
 * values can be upgraded by playing hands repeatedly, but this v0
 * implementation uses fixed Level 1 values.
 *
 * Scoring Table (Level 1):
 *   | Hand Type        | Base Chips | Multiplier | Example Score |
 *   |------------------|------------|------------|---------------|
 *   | High Card        | 5          | 1          | (5+11)×1 = 16 |
 *   | Pair             | 10         | 2          | (10+20)×2 = 60|
 *   | Two Pair         | 20         | 2          | (20+40)×2 = 120|
 *   | Three of a Kind  | 30         | 3          | (30+30)×3 = 180|
 *   | Straight         | 30         | 4          | (30+40)×4 = 280|
 *   | Flush            | 35         | 4          | (35+50)×4 = 340|
 *   | Full House       | 40         | 4          | (40+52)×4 = 368|
 *   | Four of a Kind   | 60         | 7          | (60+40)×7 = 700|
 *   | Straight Flush   | 100        | 8          | (100+40)×8 = 1120|
 */

#include "../include/balatro/scoring.hpp"

namespace balatro {

HandScoring get_hand_scoring(HandType type) {
    // Base values from Balatro (Level 1) - see file header for table
    static const HandScoring scoring_table[] = {
        {5, 1},    // HIGH_CARD
        {10, 2},   // PAIR
        {20, 2},   // TWO_PAIR
        {30, 3},   // THREE_OF_A_KIND
        {30, 4},   // STRAIGHT
        {35, 4},   // FLUSH
        {40, 4},   // FULL_HOUSE
        {60, 7},   // FOUR_OF_A_KIND
        {100, 8}   // STRAIGHT_FLUSH
    };

    return scoring_table[static_cast<int>(type)];
}

int calculate_score(const HandEvaluation& hand_eval) {
    auto scoring = get_hand_scoring(hand_eval.type);

    // Formula: (base_chips + rank_sum) × base_mult
    int total_chips = scoring.base_chips + hand_eval.rank_sum;
    int score = total_chips * scoring.base_mult;

    return score;
}

} // namespace balatro
