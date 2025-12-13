/**
 * @file hand_eval.cpp
 * @brief Implementation of poker hand evaluation algorithms.
 *
 * Key algorithms:
 *   - evaluate_hand(): Identifies hand type from 1-5 cards
 *   - find_best_hand(): Finds optimal 5-card subset from 8 cards
 *   - Pattern detection helpers for observation features
 *
 * Hand Evaluation Algorithm:
 *   1. Count rank and suit frequencies
 *   2. Check for flush (all same suit) and straight (5 consecutive)
 *   3. Match against hand types in descending strength order
 *   4. Return hand type with scoring cards and rank sum
 *
 * Straight Detection:
 *   - Normal straights: 5 consecutive ranks (e.g., 5-6-7-8-9)
 *   - Wheel straight: A-2-3-4-5 where Ace acts as low card
 *   - The wheel check looks for ranks {0,1,2,3,12} = {2,3,4,5,A}
 *
 * Scoring Cards:
 *   - Different hand types score different subsets of played cards
 *   - Pair: only the 2 paired cards (not kickers)
 *   - Full House: all 5 cards (3+2)
 *   - High Card: only the single highest card
 */

#include "../include/balatro/hand_eval.hpp"
#include <algorithm>
#include <functional>

namespace balatro {

const char* hand_type_name(HandType type) {
    static const char* names[] = {
        "High Card", "Pair", "Two Pair", "Three of a Kind",
        "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"
    };
    return names[static_cast<int>(type)];
}

namespace {

// Helper: count occurrences of each rank
std::array<int, NUM_RANKS> count_ranks(const std::vector<Card>& cards) {
    std::array<int, NUM_RANKS> counts = {};
    for (Card c : cards) {
        counts[get_rank(c)]++;
    }
    return counts;
}

// Helper: count occurrences of each suit
std::array<int, NUM_SUITS> count_suits(const std::vector<Card>& cards) {
    std::array<int, NUM_SUITS> counts = {};
    for (Card c : cards) {
        counts[get_suit(c)]++;
    }
    return counts;
}

// Check if cards form a straight (must be exactly 5 cards)
bool is_straight(const std::vector<Card>& cards) {
    if (cards.size() != 5) return false;

    std::vector<int> ranks;
    for (Card c : cards) {
        ranks.push_back(get_rank(c));
    }
    std::sort(ranks.begin(), ranks.end());

    // Check for A-2-3-4-5 (wheel)
    if (ranks[0] == 0 && ranks[1] == 1 && ranks[2] == 2 && ranks[3] == 3 && ranks[4] == 12) {
        return true;
    }

    // Check for consecutive ranks
    for (int i = 1; i < 5; ++i) {
        if (ranks[i] != ranks[i-1] + 1) return false;
    }
    return true;
}

// Check if all cards have same suit
bool is_flush(const std::vector<Card>& cards) {
    if (cards.empty()) return false;
    int suit = get_suit(cards[0]);
    for (Card c : cards) {
        if (get_suit(c) != suit) return false;
    }
    return true;
}

// Calculate rank sum for scoring
int calc_rank_sum(const std::vector<Card>& cards) {
    int sum = 0;
    for (Card c : cards) {
        sum += get_rank_value(get_rank(c));
    }
    return sum;
}

} // anonymous namespace

HandEvaluation evaluate_hand(const std::vector<Card>& cards) {
    HandEvaluation result;

    if (cards.empty() || cards.size() > 5) {
        result.cards_used = cards;
        result.rank_sum = 0;
        return result; // Invalid
    }

    auto rank_counts = count_ranks(cards);

    // Find rank frequencies
    std::vector<int> counts;
    for (int count : rank_counts) {
        if (count > 0) counts.push_back(count);
    }
    std::sort(counts.rbegin(), counts.rend());

    bool straight = is_straight(cards);
    bool flush = is_flush(cards);

    // Straight Flush - all 5 cards score
    if (straight && flush && cards.size() == 5) {
        result.type = HandType::STRAIGHT_FLUSH;
        result.cards_used = cards;
        result.rank_sum = calc_rank_sum(cards);
        return result;
    }

    // Four of a Kind - only the 4 cards score
    if (counts.size() >= 1 && counts[0] == 4) {
        result.type = HandType::FOUR_OF_A_KIND;
        // Find the rank that appears 4 times
        int quad_rank = -1;
        for (int r = 0; r < NUM_RANKS; ++r) {
            if (rank_counts[r] == 4) {
                quad_rank = r;
                break;
            }
        }
        // Add only the 4 matching cards
        for (Card c : cards) {
            if (get_rank(c) == quad_rank) {
                result.cards_used.push_back(c);
            }
        }
        result.rank_sum = calc_rank_sum(result.cards_used);
        return result;
    }

    // Full House (3 + 2) - all 5 cards score
    if (counts.size() >= 2 && counts[0] == 3 && counts[1] == 2) {
        result.type = HandType::FULL_HOUSE;
        result.cards_used = cards;
        result.rank_sum = calc_rank_sum(cards);
        return result;
    }

    // Flush - all 5 cards score
    if (flush && cards.size() == 5) {
        result.type = HandType::FLUSH;
        result.cards_used = cards;
        result.rank_sum = calc_rank_sum(cards);
        return result;
    }

    // Straight - all 5 cards score
    if (straight) {
        result.type = HandType::STRAIGHT;
        result.cards_used = cards;
        result.rank_sum = calc_rank_sum(cards);
        return result;
    }

    // Three of a Kind - only the 3 cards score
    if (counts.size() >= 1 && counts[0] == 3) {
        result.type = HandType::THREE_OF_A_KIND;
        // Find the rank that appears 3 times
        int trips_rank = -1;
        for (int r = 0; r < NUM_RANKS; ++r) {
            if (rank_counts[r] == 3) {
                trips_rank = r;
                break;
            }
        }
        // Add only the 3 matching cards
        for (Card c : cards) {
            if (get_rank(c) == trips_rank) {
                result.cards_used.push_back(c);
            }
        }
        result.rank_sum = calc_rank_sum(result.cards_used);
        return result;
    }

    // Two Pair - only the 4 cards in the pairs score
    if (counts.size() >= 2 && counts[0] == 2 && counts[1] == 2) {
        result.type = HandType::TWO_PAIR;
        // Find the two ranks that appear twice
        std::vector<int> pair_ranks;
        for (int r = 0; r < NUM_RANKS; ++r) {
            if (rank_counts[r] == 2) {
                pair_ranks.push_back(r);
            }
        }
        // Add only the cards in the pairs
        for (Card c : cards) {
            int rank = get_rank(c);
            if (std::find(pair_ranks.begin(), pair_ranks.end(), rank) != pair_ranks.end()) {
                result.cards_used.push_back(c);
            }
        }
        result.rank_sum = calc_rank_sum(result.cards_used);
        return result;
    }

    // Pair - only the 2 cards in the pair score
    if (counts.size() >= 1 && counts[0] == 2) {
        result.type = HandType::PAIR;
        // Find the rank that appears twice
        int pair_rank = -1;
        for (int r = 0; r < NUM_RANKS; ++r) {
            if (rank_counts[r] == 2) {
                pair_rank = r;
                break;
            }
        }
        // Add only the 2 matching cards
        for (Card c : cards) {
            if (get_rank(c) == pair_rank) {
                result.cards_used.push_back(c);
            }
        }
        result.rank_sum = calc_rank_sum(result.cards_used);
        return result;
    }

    // High Card - only the highest card scores
    result.type = HandType::HIGH_CARD;
    // Find the highest ranked card
    Card highest = cards[0];
    for (Card c : cards) {
        if (get_rank(c) > get_rank(highest)) {
            highest = c;
        }
    }
    result.cards_used.push_back(highest);
    result.rank_sum = calc_rank_sum(result.cards_used);
    return result;
}

HandEvaluation find_best_hand(const std::vector<Card>& cards) {
    if (cards.size() <= 5) {
        return evaluate_hand(cards);
    }

    // Try all 5-card combinations
    HandEvaluation best;
    best.type = HandType::HIGH_CARD;

    int n = static_cast<int>(cards.size());
    std::vector<int> indices(5);

    // Generate all C(n,5) combinations
    std::function<void(int, int)> generate = [&](int pos, int start) {
        if (pos == 5) {
            std::vector<Card> subset;
            for (int idx : indices) {
                subset.push_back(cards[idx]);
            }
            auto eval = evaluate_hand(subset);
            if (static_cast<int>(eval.type) > static_cast<int>(best.type)) {
                best = eval;
            }
            return;
        }

        for (int i = start; i <= n - (5 - pos); ++i) {
            indices[pos] = i;
            generate(pos + 1, i + 1);
        }
    };

    generate(0, 0);
    return best;
}

bool has_pair(const std::array<Card, HAND_SIZE>& hand) {
    auto counts = count_ranks(std::vector<Card>(hand.begin(), hand.end()));
    for (int count : counts) {
        if (count >= 2) return true;
    }
    return false;
}

bool has_three_of_kind(const std::array<Card, HAND_SIZE>& hand) {
    auto counts = count_ranks(std::vector<Card>(hand.begin(), hand.end()));
    for (int count : counts) {
        if (count >= 3) return true;
    }
    return false;
}

bool has_straight_potential(const std::array<Card, HAND_SIZE>& hand) {
    std::vector<int> ranks;
    for (Card c : hand) {
        ranks.push_back(get_rank(c));
    }
    std::sort(ranks.begin(), ranks.end());
    ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());

    // Check if we have 4 consecutive ranks
    for (size_t i = 0; i + 3 < ranks.size(); ++i) {
        bool consecutive = true;
        for (size_t j = 0; j < 3; ++j) {
            if (ranks[i + j + 1] != ranks[i + j] + 1) {
                consecutive = false;
                break;
            }
        }
        if (consecutive) return true;
    }

    // Check for A-2-3-4 (wheel potential)
    if (ranks.size() >= 4) {
        bool has_ace = (ranks.back() == 12);
        bool has_low = (ranks[0] == 0 && ranks[1] == 1 && ranks[2] == 2);
        if (has_ace && has_low) return true;
    }

    return false;
}

bool has_flush_potential(const std::array<Card, HAND_SIZE>& hand) {
    auto counts = count_suits(std::vector<Card>(hand.begin(), hand.end()));
    for (int count : counts) {
        if (count >= 4) return true;
    }
    return false;
}

int get_max_rank_bucket(const std::array<Card, HAND_SIZE>& hand) {
    int max_rank = 0;
    for (Card c : hand) {
        max_rank = std::max(max_rank, get_rank(c));
    }

    // 0: 2-4, 1: 5-7, 2: 8-10, 3: J-Q, 4: K, 5: A
    if (max_rank <= 2) return 0;  // 2-4
    if (max_rank <= 5) return 1;  // 5-7
    if (max_rank <= 8) return 2;  // 8-10
    if (max_rank <= 10) return 3; // J-Q
    if (max_rank == 11) return 4; // K
    return 5;                      // A
}

} // namespace balatro
