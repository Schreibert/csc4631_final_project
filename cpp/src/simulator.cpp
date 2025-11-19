#include "balatro/simulator.hpp"
#include "balatro/hand_eval.hpp"
#include "balatro/scoring.hpp"
#include <algorithm>
#include <functional>
#include <random>

namespace balatro {

Simulator::Simulator() {}

Observation Simulator::reset(int target_score, uint64_t seed) {
    state_.reset(target_score, seed);
    action_rng_.seed(seed + 1); // Offset to avoid same RNG state
    return state_.get_observation();
}

StepBatchResult Simulator::step_batch(const std::vector<int>& actions) {
    StepBatchResult result;
    result.rewards.resize(actions.size(), 0);
    result.done = false;
    result.win = false;

    for (size_t i = 0; i < actions.size(); ++i) {
        if (state_.is_done()) {
            // Already done - pad with zeros
            result.rewards[i] = 0;
            continue;
        }

        // Execute action
        int reward = execute_action(actions[i]);
        result.rewards[i] = reward;
    }

    result.final_obs = state_.get_observation();
    result.done = state_.is_done();
    result.win = state_.is_win();

    return result;
}

int Simulator::execute_action(int action_code) {
    std::vector<int> indices;

    switch (action_code) {
        case ACTION_PLAY_BEST:
            indices = action_play_best();
            if (!indices.empty()) {
                return state_.play_hand(indices);
            }
            break;

        case ACTION_PLAY_PAIR:
            indices = action_play_pair();
            if (!indices.empty()) {
                return state_.play_hand(indices);
            }
            break;

        case ACTION_DISCARD_NON_PAIRED:
            indices = action_discard_non_paired();
            if (!indices.empty()) {
                state_.discard_cards(indices);
            }
            break;

        case ACTION_DISCARD_LOWEST_3:
            indices = action_discard_lowest_3();
            if (!indices.empty()) {
                state_.discard_cards(indices);
            }
            break;

        case ACTION_PLAY_HIGHEST_3:
            indices = action_play_highest_3();
            if (!indices.empty()) {
                return state_.play_hand(indices);
            }
            break;

        case ACTION_RANDOM_PLAY:
            indices = action_random_play();
            if (!indices.empty()) {
                return state_.play_hand(indices);
            }
            break;

        case ACTION_RANDOM_DISCARD:
            indices = action_random_discard();
            if (!indices.empty()) {
                state_.discard_cards(indices);
            }
            break;

        default:
            // Invalid action - do nothing
            break;
    }

    return 0; // No reward for discards or invalid actions
}

std::vector<int> Simulator::action_play_best() {
    const auto& hand = state_.get_hand();
    std::vector<Card> cards(hand.begin(), hand.end());

    // Try all 5-card combinations and find best
    HandEvaluation best_eval;
    std::vector<int> best_indices;
    int best_score = -1;

    // Generate all combinations of 1-5 cards
    for (int size = 1; size <= 5; ++size) {
        std::vector<int> indices(size);

        std::function<void(int, int)> generate = [&](int pos, int start) {
            if (pos == size) {
                std::vector<Card> subset;
                for (int idx : indices) {
                    subset.push_back(hand[idx]);
                }

                auto eval = evaluate_hand(subset);
                int score = calculate_score(eval);

                if (score > best_score) {
                    best_score = score;
                    best_eval = eval;
                    best_indices = indices;
                }
                return;
            }

            for (int i = start; i <= HAND_SIZE - (size - pos); ++i) {
                indices[pos] = i;
                generate(pos + 1, i + 1);
            }
        };

        generate(0, 0);
    }

    return best_indices;
}

std::vector<int> Simulator::action_play_pair() {
    const auto& hand = state_.get_hand();

    // Count ranks
    std::array<std::vector<int>, NUM_RANKS> rank_positions;
    for (int i = 0; i < HAND_SIZE; ++i) {
        int rank = get_rank(hand[i]);
        rank_positions[rank].push_back(i);
    }

    // Find highest pair
    for (int rank = NUM_RANKS - 1; rank >= 0; --rank) {
        if (rank_positions[rank].size() >= 2) {
            return {rank_positions[rank][0], rank_positions[rank][1]};
        }
    }

    return {}; // No pair found
}

std::vector<int> Simulator::action_discard_non_paired() {
    const auto& hand = state_.get_hand();

    // Count ranks
    std::array<int, NUM_RANKS> rank_counts = {};
    for (Card c : hand) {
        rank_counts[get_rank(c)]++;
    }

    // Collect indices of non-paired cards
    std::vector<int> indices;
    for (int i = 0; i < HAND_SIZE; ++i) {
        int rank = get_rank(hand[i]);
        if (rank_counts[rank] == 1) {
            indices.push_back(i);
        }
    }

    return indices;
}

std::vector<int> Simulator::action_discard_lowest_3() {
    const auto& hand = state_.get_hand();

    // Create pairs of (rank, index)
    std::vector<std::pair<int, int>> rank_idx;
    for (int i = 0; i < HAND_SIZE; ++i) {
        rank_idx.push_back({get_rank(hand[i]), i});
    }

    // Sort by rank
    std::sort(rank_idx.begin(), rank_idx.end());

    // Take lowest 3
    std::vector<int> indices;
    for (int i = 0; i < std::min(3, HAND_SIZE); ++i) {
        indices.push_back(rank_idx[i].second);
    }

    return indices;
}

std::vector<int> Simulator::action_play_highest_3() {
    const auto& hand = state_.get_hand();

    // Create pairs of (rank, index)
    std::vector<std::pair<int, int>> rank_idx;
    for (int i = 0; i < HAND_SIZE; ++i) {
        rank_idx.push_back({get_rank(hand[i]), i});
    }

    // Sort by rank descending
    std::sort(rank_idx.rbegin(), rank_idx.rend());

    // Take highest 3
    std::vector<int> indices;
    for (int i = 0; i < std::min(3, HAND_SIZE); ++i) {
        indices.push_back(rank_idx[i].second);
    }

    return indices;
}

std::vector<int> Simulator::action_random_play() {
    // Random number of cards (1-5)
    std::uniform_int_distribution<int> size_dist(1, 5);
    int num_cards = size_dist(action_rng_);

    // Random selection
    std::vector<int> all_indices;
    for (int i = 0; i < HAND_SIZE; ++i) {
        all_indices.push_back(i);
    }
    std::shuffle(all_indices.begin(), all_indices.end(), action_rng_);

    std::vector<int> indices(all_indices.begin(), all_indices.begin() + num_cards);
    return indices;
}

std::vector<int> Simulator::action_random_discard() {
    // Random number of cards (1-8)
    std::uniform_int_distribution<int> size_dist(1, HAND_SIZE);
    int num_cards = size_dist(action_rng_);

    // Random selection
    std::vector<int> all_indices;
    for (int i = 0; i < HAND_SIZE; ++i) {
        all_indices.push_back(i);
    }
    std::shuffle(all_indices.begin(), all_indices.end(), action_rng_);

    std::vector<int> indices(all_indices.begin(), all_indices.begin() + num_cards);
    return indices;
}

} // namespace balatro
