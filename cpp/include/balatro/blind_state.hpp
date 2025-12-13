/**
 * @file blind_state.hpp
 * @brief Single-blind episode state management and RL action interface.
 *
 * Implements the core game loop for a single blind attempt (one RL episode).
 * Manages resources (plays/discards), hand state, score tracking, and win/loss.
 *
 * Episode Rules (v0 Simplified Ruleset):
 *   - Start: 4 plays, 3 discards, 8-card hand, shuffled 52-card deck
 *   - Win condition: chips >= target_score at any moment
 *   - Loss condition: plays_left == 0 AND chips < target_score
 *   - Discarding draws replacement cards from deck
 *   - Playing does NOT consume cards (play from same 8-card hand)
 *
 * RL Interface:
 *   - Observation: 24-feature struct with state, hand analysis, and card data
 *   - Action: Type (PLAY/DISCARD) + 8-element boolean card mask
 *   - Rewards: Raw chip delta (reward shaping done in Python layer)
 *
 * RL Helper Methods:
 *   - get_best_hand(): Pre-computed best 5-card hand analysis
 *   - predict_play_score(): Evaluate action without executing
 *   - enumerate_all_actions(): Generate all valid actions (~450-500 total)
 */

#pragma once

#include "card.hpp"
#include "hand_eval.hpp"
#include <cstdint>
#include <array>

namespace balatro {

/**
 * @brief Complete game state observation for RL agents.
 *
 * Provides both raw state (resources, chips) and derived features
 * (hand analysis, pattern detection) to help agents make decisions.
 * All fields are integers or booleans for easy tensor conversion.
 */
struct Observation {
    // Resource State (4 features)
    int plays_left;         ///< Remaining plays (0-4, starts at 4)
    int discards_left;      ///< Remaining discards (0-3, starts at 3)
    int chips;              ///< Current chips accumulated this episode
    int chips_to_target;    ///< Chips needed to win: max(target - chips, 0)

    // Deck State (1 feature)
    int deck_remaining;     ///< Cards left in draw pile (0-44 after initial deal)

    // Hand Composition (2 features)
    int num_face_cards;     ///< Count of J/Q/K in hand (0-8)
    int num_aces;           ///< Count of Aces in hand (0-8)

    // Hand Pattern Potential (4 boolean features)
    bool has_pair;          ///< True if any two cards share rank
    bool has_trips;         ///< True if any three cards share rank
    bool straight_potential;///< True if one card away from straight
    bool flush_potential;   ///< True if 4+ cards share suit

    // Best Hand Analysis (2 features) - Pre-computed by C++
    int best_hand_type;     ///< HandType enum (0-8) for best 5-card hand
    int best_hand_score;    ///< Predicted chips if best hand is played

    // Complete Hand Patterns (6 boolean features)
    bool has_two_pair;      ///< Best hand is Two Pair or better
    bool has_full_house;    ///< Best hand is Full House
    bool has_four_of_kind;  ///< Best hand is Four of a Kind
    bool has_straight;      ///< Best hand is Straight (or Straight Flush)
    bool has_flush;         ///< Best hand is Flush (or Straight Flush)
    bool has_straight_flush;///< Best hand is Straight Flush

    // Current Hand Cards (16 features: 8 ranks + 8 suits)
    int card_ranks[HAND_SIZE];  ///< Rank of each card (0-12, where 0=2, 12=A)
    int card_suits[HAND_SIZE];  ///< Suit of each card (0-3: C/D/H/S)
};

/**
 * @brief Agent action with direct card selection via boolean mask.
 *
 * Actions specify whether to PLAY or DISCARD, plus which cards to select.
 * The card_mask is an 8-element boolean array where true = selected.
 *
 * Constraints:
 *   - PLAY: Must select 1-5 cards (selected cards form poker hand)
 *   - DISCARD: Must select 1-5 cards (selected cards are replaced)
 *
 * Example:
 * @code
 *   Action a;
 *   a.type = Action::PLAY;
 *   a.card_mask = {true, true, true, true, true, false, false, false};
 *   // Plays first 5 cards in hand
 * @endcode
 */
struct Action {
    /** @brief Action type: PLAY (0) executes hand, DISCARD (1) replaces cards */
    enum Type {
        PLAY = 0,    ///< Play selected cards as poker hand (scores chips)
        DISCARD = 1  ///< Discard selected cards and draw replacements
    };

    Type type;                              ///< PLAY or DISCARD
    std::array<bool, HAND_SIZE> card_mask;  ///< True for each card to select

    /** @brief Default constructor: PLAY with no cards selected */
    Action() : type(PLAY), card_mask{} {}

    /**
     * @brief Construct action with type and card mask.
     * @param t Action type (PLAY or DISCARD)
     * @param mask 8-element boolean mask for card selection
     */
    Action(Type t, const std::array<bool, HAND_SIZE>& mask) : type(t), card_mask(mask) {}
};

/**
 * @brief Result of evaluating an action without executing it.
 *
 * Used by enumerate_all_actions() to provide agents with action outcomes
 * for planning without trial-and-error simulation.
 */
struct ActionOutcome {
    Action action;           ///< The action being evaluated
    bool valid;              ///< True if action is legal in current state
    int predicted_chips;     ///< For PLAY: predicted chip score; for DISCARD: 0
    int predicted_hand_type; ///< For PLAY: HandType enum (0-8); for DISCARD: 0

    /** @brief Default constructor: invalid action with zero predictions */
    ActionOutcome() : valid(false), predicted_chips(0), predicted_hand_type(0) {}
};

/**
 * @brief Manages a single blind attempt (one RL episode).
 *
 * Handles all game logic for one episode:
 *   - Deterministic deck shuffling based on seed
 *   - Resource tracking (plays/discards remaining)
 *   - Score accumulation and win/loss detection
 *   - Hand management and card replacement
 *
 * Episode Lifecycle:
 *   1. reset() - Initialize with target score and seed
 *   2. Loop until is_done():
 *      - get_observation() - Get current state
 *      - play_hand() or discard_cards() - Execute action
 *   3. is_win() - Check outcome
 *
 * Thread Safety: Not thread-safe. Each thread should have its own instance.
 */
class BlindState {
public:
    /** @brief Construct uninitialized state (call reset() before use) */
    BlindState();

    /**
     * @brief Initialize a new episode with target score and random seed.
     * @param target_score Chips needed to win (e.g., 300)
     * @param seed Random seed for deterministic deck shuffling
     *
     * Initializes: 4 plays, 3 discards, shuffled deck, dealt 8-card hand, 0 chips.
     * Same (target_score, seed) always produces identical starting state.
     */
    void reset(int target_score, uint64_t seed);

    /**
     * @brief Build current observation for RL agent.
     * @return Observation struct with all 24 features populated
     *
     * Computes derived features (best hand, patterns) on each call.
     * Relatively expensive due to hand evaluation (~56 combinations).
     */
    Observation get_observation() const;

    /** @brief Check if episode has ended (win or loss) */
    bool is_done() const { return done_; }

    /** @brief Check if episode was won (chips >= target reached) */
    bool is_win() const { return win_; }

    /** @brief Get current accumulated chips */
    int get_chips() const { return chips_; }

    /** @brief Get target score needed to win */
    int get_target_score() const { return target_score_; }

    /** @brief Get remaining plays (0-4) */
    int get_plays_left() const { return plays_left_; }

    /** @brief Get remaining discards (0-3) */
    int get_discards_left() const { return discards_left_; }

    /** @brief Get current 8-card hand (read-only) */
    const Hand& get_hand() const { return hand_; }

    /** @brief Get deck state (read-only, for debugging) */
    const Deck& get_deck() const { return deck_; }

    /**
     * @brief Execute a PLAY action and score the selected cards.
     * @param action Action with type=PLAY and card_mask indicating selection
     * @return Chip delta (score added), or 0 if invalid action
     *
     * Decrements plays_left, adds chips, checks for win condition.
     * Does NOT replace cards - hand remains the same after play.
     *
     * @note Cards are NOT consumed. You can play different subsets
     *       of the same 8-card hand on each play.
     */
    int play_hand(const Action& action);

    /**
     * @brief Execute a DISCARD action and draw replacement cards.
     * @param action Action with type=DISCARD and card_mask indicating selection
     *
     * Decrements discards_left, discards selected cards to pile,
     * draws replacements from deck. Hand size remains 8.
     */
    void discard_cards(const Action& action);

    /**
     * @brief Check if an action is valid in current state.
     * @param action Action to validate
     * @return True if action can be executed
     *
     * Checks: correct type, resources available, valid card count (1-5).
     */
    bool is_valid_action(const Action& action) const;

    // RL Helper Methods
    // These methods help agents make decisions without trial-and-error.

    /**
     * @brief Get the best possible 5-card poker hand from current hand.
     * @return HandEvaluation with type, cards, and rank_sum
     *
     * Searches all C(8,5) = 56 combinations to find strongest hand.
     * Use this to know hand strength without exhaustive search in Python.
     */
    HandEvaluation get_best_hand() const;

    /**
     * @brief Predict chip score for a PLAY action without executing it.
     * @param card_mask Boolean mask indicating which cards to play
     * @return Predicted chip score, or 0 if selection is invalid
     *
     * Allows agents to evaluate actions before committing.
     * Does not modify game state.
     */
    int predict_play_score(const std::array<bool, HAND_SIZE>& card_mask) const;

    /**
     * @brief Enumerate all valid actions with predicted outcomes.
     * @return Vector of ActionOutcome sorted by predicted_chips (descending)
     *
     * Generates:
     *   - All valid PLAY actions (C(8,1) + C(8,2) + ... + C(8,5) = 218 combos)
     *   - All valid DISCARD actions (limited to ~250 representative samples)
     *
     * For PLAY actions, includes exact predicted_chips and predicted_hand_type.
     * For DISCARD actions, predicted values are 0 (outcome depends on draw).
     *
     * Typical return size: ~450-500 actions. Use this for value-based RL
     * methods that need to evaluate all actions.
     */
    std::vector<ActionOutcome> enumerate_all_actions() const;

private:
    /** @brief Check win/loss conditions and update done_/win_ flags */
    void check_termination();

    /**
     * @brief Convert boolean card mask to vector of indices.
     * @param mask 8-element boolean array
     * @return Vector of indices where mask[i] == true
     */
    std::vector<int> mask_to_indices(const std::array<bool, HAND_SIZE>& mask) const;

    Deck deck_;           ///< Card deck with draw pile and discards
    Hand hand_;           ///< Current 8-card hand

    int target_score_;    ///< Chips needed to win
    int chips_;           ///< Current accumulated chips
    int plays_left_;      ///< Remaining plays (0-4)
    int discards_left_;   ///< Remaining discards (0-3)

    bool done_;           ///< True when episode has ended
    bool win_;            ///< True if episode was won (only valid when done_)
};

} // namespace balatro
