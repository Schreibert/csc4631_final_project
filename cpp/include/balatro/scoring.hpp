#pragma once

#include "balatro/hand_eval.hpp"

namespace balatro {

// Base chips and multipliers for each hand type (Level 1)
struct HandScoring {
    int base_chips;
    int base_mult;
};

// Get base scoring for a hand type
HandScoring get_hand_scoring(HandType type);

// Calculate total score for a played hand
// Formula: (base_chips + rank_sum) × base_mult
int calculate_score(const HandEvaluation& hand_eval);

} // namespace balatro
