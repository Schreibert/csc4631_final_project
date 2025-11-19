#include "balatro/blind_state.hpp"
#include "balatro/scoring.hpp"
#include <algorithm>

namespace balatro {

BlindState::BlindState()
    : target_score_(0), chips_(0), plays_left_(0), discards_left_(0),
      done_(false), win_(false) {}

void BlindState::reset(int target_score, uint64_t seed) {
    target_score_ = target_score;
    chips_ = 0;
    plays_left_ = 4;
    discards_left_ = 3;
    done_ = false;
    win_ = false;

    // Initialize deck and deal hand
    deck_.reset(seed);
    deal_hand(deck_, hand_);
}

Observation BlindState::get_observation() const {
    Observation obs;

    obs[OBS_PLAYS_LEFT] = plays_left_;
    obs[OBS_DISCARDS_LEFT] = discards_left_;
    obs[OBS_CHIPS_TO_TARGET] = std::max(0, target_score_ - chips_);
    obs[OBS_HAS_PAIR] = has_pair(hand_) ? 1 : 0;
    obs[OBS_HAS_TRIPS] = has_three_of_kind(hand_) ? 1 : 0;
    obs[OBS_STRAIGHT_POT] = has_straight_potential(hand_) ? 1 : 0;
    obs[OBS_FLUSH_POT] = has_flush_potential(hand_) ? 1 : 0;
    obs[OBS_MAX_RANK_BUCKET] = get_max_rank_bucket(hand_);

    return obs;
}

int BlindState::play_hand(const std::vector<int>& card_indices) {
    if (done_ || plays_left_ <= 0) {
        return 0; // No action if done or no plays left
    }

    // Build subset of cards to play
    std::vector<Card> cards_to_play;
    for (int idx : card_indices) {
        if (idx >= 0 && idx < HAND_SIZE) {
            cards_to_play.push_back(hand_[idx]);
        }
    }

    if (cards_to_play.empty()) {
        return 0; // Invalid play
    }

    // Evaluate and score
    auto eval = evaluate_hand(cards_to_play);
    int score = calculate_score(eval);

    int old_chips = chips_;
    chips_ += score;
    plays_left_--;

    // Replace played cards
    replace_cards(deck_, hand_, card_indices);

    // Check termination
    check_termination();

    return chips_ - old_chips; // Chip delta for reward
}

void BlindState::discard_cards(const std::vector<int>& card_indices) {
    if (done_ || discards_left_ <= 0) {
        return; // No action if done or no discards left
    }

    if (card_indices.empty()) {
        return; // Invalid discard
    }

    discards_left_--;

    // Replace discarded cards
    replace_cards(deck_, hand_, card_indices);
}

void BlindState::check_termination() {
    // Win if chips >= target
    if (chips_ >= target_score_) {
        done_ = true;
        win_ = true;
        return;
    }

    // Loss if no plays left and chips < target
    if (plays_left_ <= 0 && chips_ < target_score_) {
        done_ = true;
        win_ = false;
        return;
    }
}

} // namespace balatro
