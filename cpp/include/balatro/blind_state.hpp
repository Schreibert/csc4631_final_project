#pragma once

#include "card.hpp"
#include "hand_eval.hpp"
#include <cstdint>
#include <array>

namespace balatro {

// Enhanced observation with full card visibility
struct Observation {
    // State features
    int plays_left;
    int discards_left;
    int chips;              // Current chips
    int chips_to_target;    // max(target - chips, 0)
    int deck_remaining;     // Cards left in deck
    int discard_pile_size;  // Cards in discard pile
    int num_face_cards;    // Number of face cards in hand
    int num_aces;          // Number of aces in hand

    // Hand analysis features
    bool has_pair;
    bool has_trips;
    bool straight_potential;
    bool flush_potential;

    // Best possible hand analysis
    int best_hand_type;         // 0-8 (HandType enum value)
    int best_hand_score;        // Pre-calculated chip score for best hand

    // Complete hand pattern flags (actual hands, not just potential)
    bool has_two_pair;
    bool has_full_house;
    bool has_four_of_kind;
    bool has_straight;          // Actual straight
    bool has_flush;             // Actual flush
    bool has_straight_flush;

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

// Result of action enumeration (for RL planning)
struct ActionOutcome {
    Action action;
    bool valid;
    int predicted_chips;      // For PLAY actions only
    int predicted_hand_type;  // For PLAY actions only (HandType enum value)

    ActionOutcome() : valid(false), predicted_chips(0), predicted_hand_type(0) {}
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

    // RL helper methods

    // Get the best possible hand from current hand
    HandEvaluation get_best_hand() const;

    // Predict the score for a PLAY action without executing it
    int predict_play_score(const std::array<bool, HAND_SIZE>& card_mask) const;

    // Enumerate all valid actions with predicted outcomes
    std::vector<ActionOutcome> enumerate_all_actions() const;

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
