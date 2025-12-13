/**
 * @file blind_state.cpp
 * @brief Implementation of episode state management and RL helpers.
 *
 * Implements the core game loop for single-blind Balatro poker:
 *   - Episode initialization with deterministic deck shuffling
 *   - Play and discard action execution
 *   - Win/loss condition checking
 *   - Observation construction with 24 features
 *
 * RL Helper Methods:
 *   - get_best_hand(): Finds optimal 5-card hand from 8 cards
 *   - predict_play_score(): Evaluates play action without executing
 *   - enumerate_all_actions(): Generates ~450-500 valid actions
 *
 * Action Enumeration Algorithm:
 *   - PLAY actions: C(8,1) + C(8,2) + C(8,3) + C(8,4) + C(8,5) = 218 combinations
 *   - DISCARD actions: C(8,1) + ... + C(8,5) = 218 combinations (max 5 cards)
 *   - Total: ~436 actions per state (varies if plays/discards exhausted)
 *   - Sorted by predicted_chips (descending) for PLAY actions
 */

#include "../include/balatro/blind_state.hpp"
#include "../include/balatro/scoring.hpp"
#include "../include/balatro/hand_eval.hpp"
#include <algorithm>
#include <functional>

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
    obs.num_face_cards = std::count_if(hand_.begin(), hand_.end(), [](const Card& c) {
        int rank = get_rank(c);
        return rank >= 9; // J, Q, K, A (ranks 9-12)
    });
    obs.num_aces = std::count_if(hand_.begin(), hand_.end(), [](const Card& c) {
        return get_rank(c) == 12; // Ace
    });

    // Hand analysis features
    obs.has_pair = has_pair(hand_) ? 1 : 0;
    obs.has_trips = has_three_of_kind(hand_) ? 1 : 0;
    obs.straight_potential = has_straight_potential(hand_) ? 1 : 0;
    obs.flush_potential = has_flush_potential(hand_) ? 1 : 0;

    // Best possible hand analysis (for RL agents)
    std::vector<Card> hand_vec(hand_.begin(), hand_.end());
    HandEvaluation best_hand = find_best_hand(hand_vec);
    obs.best_hand_type = static_cast<int>(best_hand.type);
    obs.best_hand_score = calculate_score(best_hand);

    // Complete hand pattern flags based on best hand
    obs.has_two_pair = (best_hand.type == HandType::TWO_PAIR);
    obs.has_full_house = (best_hand.type == HandType::FULL_HOUSE);
    obs.has_four_of_kind = (best_hand.type == HandType::FOUR_OF_A_KIND);
    obs.has_straight = (best_hand.type == HandType::STRAIGHT || best_hand.type == HandType::STRAIGHT_FLUSH);
    obs.has_flush = (best_hand.type == HandType::FLUSH || best_hand.type == HandType::STRAIGHT_FLUSH);
    obs.has_straight_flush = (best_hand.type == HandType::STRAIGHT_FLUSH);

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

    // If discarding, must select at most 5 cards
    if (action.type == Action::DISCARD && count > 5) {
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

// RL helper methods

HandEvaluation BlindState::get_best_hand() const {
    std::vector<Card> hand_vec(hand_.begin(), hand_.end());
    return find_best_hand(hand_vec);
}

int BlindState::predict_play_score(const std::array<bool, HAND_SIZE>& card_mask) const {
    // Convert mask to card vector
    std::vector<Card> cards;
    for (int i = 0; i < HAND_SIZE; ++i) {
        if (card_mask[i]) {
            cards.push_back(hand_[i]);
        }
    }

    // Evaluate and score
    if (cards.empty()) {
        return 0;
    }

    auto eval = evaluate_hand(cards);
    return calculate_score(eval);
}

std::vector<ActionOutcome> BlindState::enumerate_all_actions() const {
    std::vector<ActionOutcome> outcomes;

    // Helper lambda to generate all k-element subsets (combinations)
    auto generate_combinations = [](int n, int k) {
        std::vector<std::array<bool, HAND_SIZE>> masks;
        std::array<bool, HAND_SIZE> mask{};

        // Recursive generator using backtracking
        std::function<void(int, int)> generate = [&](int start, int count) {
            if (count == k) {
                masks.push_back(mask);
                return;
            }
            for (int i = start; i <= n - (k - count); ++i) {
                mask[i] = true;
                generate(i + 1, count + 1);
                mask[i] = false;
            }
        };

        generate(0, 0);
        return masks;
    };

    // Generate all PLAY actions (1-5 cards)
    if (plays_left_ > 0) {
        for (int num_cards = 1; num_cards <= std::min(5, HAND_SIZE); ++num_cards) {
            auto masks = generate_combinations(HAND_SIZE, num_cards);
            for (const auto& mask : masks) {
                ActionOutcome outcome;
                outcome.action = Action(Action::PLAY, mask);
                outcome.valid = true;
                outcome.predicted_chips = predict_play_score(mask);

                // Get hand type
                std::vector<Card> cards;
                for (int i = 0; i < HAND_SIZE; ++i) {
                    if (mask[i]) cards.push_back(hand_[i]);
                }
                auto eval = evaluate_hand(cards);
                outcome.predicted_hand_type = static_cast<int>(eval.type);

                outcomes.push_back(outcome);
            }
        }
    }

    // Generate all DISCARD actions (1-5 cards, same limit as PLAY)
    if (discards_left_ > 0) {
        for (int num_cards = 1; num_cards <= std::min(5, HAND_SIZE); ++num_cards) {
            auto masks = generate_combinations(HAND_SIZE, num_cards);
            for (const auto& mask : masks) {
                ActionOutcome outcome;
                outcome.action = Action(Action::DISCARD, mask);
                outcome.valid = true;
                outcome.predicted_chips = 0;  // Discarding doesn't score chips
                outcome.predicted_hand_type = 0;  // Not applicable for discards

                outcomes.push_back(outcome);
            }
        }
    }

    // Sort by predicted chips (descending) for PLAY actions
    std::sort(outcomes.begin(), outcomes.end(), [](const ActionOutcome& a, const ActionOutcome& b) {
        if (a.action.type == Action::PLAY && b.action.type == Action::PLAY) {
            return a.predicted_chips > b.predicted_chips;
        }
        return a.action.type < b.action.type;  // PLAYs before DISCARDs
    });

    return outcomes;
}

} // namespace balatro
