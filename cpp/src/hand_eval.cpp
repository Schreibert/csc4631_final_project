#include "balatro/hand_eval.hpp"
#include <algorithm>
#include <functional>
#include <unordered_map>

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
    result.cards_used = cards;
    result.rank_sum = calc_rank_sum(cards);

    if (cards.empty() || cards.size() > 5) {
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

    // Straight Flush
    if (straight && flush && cards.size() == 5) {
        result.type = HandType::STRAIGHT_FLUSH;
        return result;
    }

    // Four of a Kind
    if (counts.size() >= 1 && counts[0] == 4) {
        result.type = HandType::FOUR_OF_A_KIND;
        return result;
    }

    // Full House (3 + 2)
    if (counts.size() >= 2 && counts[0] == 3 && counts[1] == 2) {
        result.type = HandType::FULL_HOUSE;
        return result;
    }

    // Flush
    if (flush && cards.size() == 5) {
        result.type = HandType::FLUSH;
        return result;
    }

    // Straight
    if (straight) {
        result.type = HandType::STRAIGHT;
        return result;
    }

    // Three of a Kind
    if (counts.size() >= 1 && counts[0] == 3) {
        result.type = HandType::THREE_OF_A_KIND;
        return result;
    }

    // Two Pair
    if (counts.size() >= 2 && counts[0] == 2 && counts[1] == 2) {
        result.type = HandType::TWO_PAIR;
        return result;
    }

    // Pair
    if (counts.size() >= 1 && counts[0] == 2) {
        result.type = HandType::PAIR;
        return result;
    }

    // High Card
    result.type = HandType::HIGH_CARD;
    return result;
}

HandEvaluation find_best_hand(const std::vector<Card>& cards) {
    if (cards.size() <= 5) {
        return evaluate_hand(cards);
    }

    // Try all 5-card combinations
    HandEvaluation best;
    best.type = HandType::HIGH_CARD;

    int n = cards.size();
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
