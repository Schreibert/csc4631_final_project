#pragma once

#include "balatro/card.hpp"
#include "balatro/hand_eval.hpp"
#include <cstdint>
#include <array>

namespace balatro {

// Observation vector size (v0 contract)
constexpr int OBS_SIZE = 8;

// Observation vector indices
constexpr int OBS_PLAYS_LEFT = 0;
constexpr int OBS_DISCARDS_LEFT = 1;
constexpr int OBS_CHIPS_TO_TARGET = 2;
constexpr int OBS_HAS_PAIR = 3;
constexpr int OBS_HAS_TRIPS = 4;
constexpr int OBS_STRAIGHT_POT = 5;
constexpr int OBS_FLUSH_POT = 6;
constexpr int OBS_MAX_RANK_BUCKET = 7;

using Observation = std::array<int, OBS_SIZE>;

// State of a single blind (episode)
class BlindState {
public:
    BlindState();

    // Initialize a new episode
    void reset(int target_score, uint64_t seed);

    // Get current observation
    Observation get_observation() const;

    // Check if episode is done
    bool is_done() const { return done_; }

    // Check if episode was won
    bool is_win() const { return win_; }

    // Get current chips
    int get_chips() const { return chips_; }

    // Get target score
    int get_target_score() const { return target_score_; }

    // Get plays left
    int get_plays_left() const { return plays_left_; }

    // Get discards left
    int get_discards_left() const { return discards_left_; }

    // Get current hand (read-only)
    const Hand& get_hand() const { return hand_; }

    // Play a hand (returns chip delta for reward)
    int play_hand(const std::vector<int>& card_indices);

    // Discard cards
    void discard_cards(const std::vector<int>& card_indices);

private:
    void check_termination();

    Deck deck_;
    Hand hand_;

    int target_score_;
    int chips_;
    int plays_left_;
    int discards_left_;

    bool done_;
    bool win_;
};

} // namespace balatro
