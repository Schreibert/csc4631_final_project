#include "../include/balatro/blind_state.hpp"
#include "../include/balatro/scoring.hpp"
#include "../include/balatro/hand_eval.hpp"
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

    // State features
    obs.plays_left = plays_left_;
    obs.discards_left = discards_left_;
    obs.chips = chips_;
    obs.chips_to_target = std::max(0, target_score_ - chips_);
    obs.deck_remaining = deck_.remaining();
    obs.discard_pile_size = 0; // TODO: Add discard pile tracking to Deck class

    // Hand analysis features
    obs.has_pair = has_pair(hand_) ? 1 : 0;
    obs.has_trips = has_three_of_kind(hand_) ? 1 : 0;
    obs.straight_potential = has_straight_potential(hand_) ? 1 : 0;
    obs.flush_potential = has_flush_potential(hand_) ? 1 : 0;

    // Current hand cards
    for (int i = 0; i < HAND_SIZE; ++i) {
        obs.card_ranks[i] = get_rank(hand_[i]);
        obs.card_suits[i] = get_suit(hand_[i]);
    }

    return obs;
}

int BlindState::play_hand(const Action& action) {
    if (done_ || plays_left_ <= 0) {
        return 0; // No action if done or no plays left
    }

    // Validate action (should be done before calling, but double-check)
    if (!is_valid_action(action)) {
        return 0; // Invalid action
    }

    // Convert mask to indices and build card subset
    std::vector<int> card_indices = mask_to_indices(action.card_mask);
    std::vector<Card> cards_to_play;
    for (int idx : card_indices) {
        cards_to_play.push_back(hand_[idx]);
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

void BlindState::discard_cards(const Action& action) {
    if (done_ || discards_left_ <= 0) {
        return; // No action if done or no discards left
    }

    // Validate action
    if (!is_valid_action(action)) {
        return; // Invalid action
    }

    discards_left_--;

    // Convert mask to indices and replace cards
    std::vector<int> card_indices = mask_to_indices(action.card_mask);
    replace_cards(deck_, hand_, card_indices);
}

bool BlindState::is_valid_action(const Action& action) const {
    // Count selected cards
    int count = 0;
    for (bool selected : action.card_mask) {
        if (selected) count++;
    }

    // Must select at least 1 card
    if (count == 0) {
        return false;
    }

    // If playing, must select at most 5 cards
    if (action.type == Action::PLAY && count > 5) {
        return false;
    }

    // If playing, must have plays left
    if (action.type == Action::PLAY && plays_left_ <= 0) {
        return false;
    }

    // If discarding, must have discards left
    if (action.type == Action::DISCARD && discards_left_ <= 0) {
        return false;
    }

    // Can't take action if episode is done
    if (done_) {
        return false;
    }

    return true;
}

std::vector<int> BlindState::mask_to_indices(const std::array<bool, HAND_SIZE>& mask) const {
    std::vector<int> indices;
    for (int i = 0; i < HAND_SIZE; ++i) {
        if (mask[i]) {
            indices.push_back(i);
        }
    }
    return indices;
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
