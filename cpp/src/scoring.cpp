#include "balatro/scoring.hpp"

namespace balatro {

HandScoring get_hand_scoring(HandType type) {
    // Base values from Balatro (Level 1)
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
