#pragma once

#include "card.hpp"
#include <cstdint>
#include <array>

namespace balatro {

// Enhanced observation with full card visibility
struct Observation {
    // State features (8 values)
    int plays_left;
    int discards_left;
    int chips;              // Current chips
    int chips_to_target;    // max(target - chips, 0)
    int deck_remaining;     // Cards left in deck
    int discard_pile_size;  // Cards in discard pile

    // Hand analysis features (4 values)
    bool has_pair;           
    bool has_trips;          
    bool straight_potential; 
    bool flush_potential;    

    // Current hand - 8 cards (16 values: rank, suit for each)
    int card_ranks[HAND_SIZE];  // 0-12 (2 through Ace)
    int card_suits[HAND_SIZE];  // 0-3 (Clubs, Diamonds, Hearts, Spades)
};

// Action structure for direct card selection
struct Action {
    enum Type {
        PLAY = 0,
        DISCARD = 1
    };

    Type type;
    std::array<bool, HAND_SIZE> card_mask;  // Which cards to select

    // Constructor for convenience
    Action() : type(PLAY), card_mask{} {}
    Action(Type t, const std::array<bool, HAND_SIZE>& mask) : type(t), card_mask(mask) {}
};

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

    // Get deck (read-only)
    const Deck& get_deck() const { return deck_; }

    // Play a hand with direct card selection (returns chip delta for reward)
    int play_hand(const Action& action);

    // Discard cards with direct card selection
    void discard_cards(const Action& action);

    // Validate an action
    bool is_valid_action(const Action& action) const;

private:
    void check_termination();

    // Helper: Convert card mask to indices
    std::vector<int> mask_to_indices(const std::array<bool, HAND_SIZE>& mask) const;

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
