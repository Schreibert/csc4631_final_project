#include <gtest/gtest.h>
#include "balatro/hand_eval.hpp"

using namespace balatro;

TEST(HandEvalTest, HighCard) {
    // 2♣ 5♦ 7♥ 9♠ K♣
    std::vector<Card> cards = {
        make_card(0, 0),  // 2♣
        make_card(3, 1),  // 5♦
        make_card(5, 2),  // 7♥
        make_card(7, 3),  // 9♠
        make_card(11, 0)  // K♣
    };

    auto eval = evaluate_hand(cards);
    EXPECT_EQ(eval.type, HandType::HIGH_CARD);
}

TEST(HandEvalTest, Pair) {
    // K♣ K♦ 2♥ 5♠ 9♣
    std::vector<Card> cards = {
        make_card(11, 0), // K♣
        make_card(11, 1), // K♦
        make_card(0, 2),  // 2♥
        make_card(3, 3),  // 5♠
        make_card(7, 0)   // 9♣
    };

    auto eval = evaluate_hand(cards);
    EXPECT_EQ(eval.type, HandType::PAIR);
}

TEST(HandEvalTest, TwoPair) {
    // K♣ K♦ 5♥ 5♠ 2♣
    std::vector<Card> cards = {
        make_card(11, 0), // K♣
        make_card(11, 1), // K♦
        make_card(3, 2),  // 5♥
        make_card(3, 3),  // 5♠
        make_card(0, 0)   // 2♣
    };

    auto eval = evaluate_hand(cards);
    EXPECT_EQ(eval.type, HandType::TWO_PAIR);
}

TEST(HandEvalTest, ThreeOfAKind) {
    // K♣ K♦ K♥ 2♠ 5♣
    std::vector<Card> cards = {
        make_card(11, 0), // K♣
        make_card(11, 1), // K♦
        make_card(11, 2), // K♥
        make_card(0, 3),  // 2♠
        make_card(3, 0)   // 5♣
    };

    auto eval = evaluate_hand(cards);
    EXPECT_EQ(eval.type, HandType::THREE_OF_A_KIND);
}

TEST(HandEvalTest, Straight) {
    // 5♣ 6♦ 7♥ 8♠ 9♣
    std::vector<Card> cards = {
        make_card(3, 0),  // 5♣
        make_card(4, 1),  // 6♦
        make_card(5, 2),  // 7♥
        make_card(6, 3),  // 8♠
        make_card(7, 0)   // 9♣
    };

    auto eval = evaluate_hand(cards);
    EXPECT_EQ(eval.type, HandType::STRAIGHT);
}

TEST(HandEvalTest, StraightWheel) {
    // A♣ 2♦ 3♥ 4♠ 5♣ (wheel)
    std::vector<Card> cards = {
        make_card(12, 0), // A♣
        make_card(0, 1),  // 2♦
        make_card(1, 2),  // 3♥
        make_card(2, 3),  // 4♠
        make_card(3, 0)   // 5♣
    };

    auto eval = evaluate_hand(cards);
    EXPECT_EQ(eval.type, HandType::STRAIGHT);
}

TEST(HandEvalTest, Flush) {
    // 2♣ 5♣ 7♣ 9♣ K♣ (all clubs)
    std::vector<Card> cards = {
        make_card(0, 0),  // 2♣
        make_card(3, 0),  // 5♣
        make_card(5, 0),  // 7♣
        make_card(7, 0),  // 9♣
        make_card(11, 0)  // K♣
    };

    auto eval = evaluate_hand(cards);
    EXPECT_EQ(eval.type, HandType::FLUSH);
}

TEST(HandEvalTest, FullHouse) {
    // K♣ K♦ K♥ A♠ A♣
    std::vector<Card> cards = {
        make_card(11, 0), // K♣
        make_card(11, 1), // K♦
        make_card(11, 2), // K♥
        make_card(12, 3), // A♠
        make_card(12, 0)  // A♣
    };

    auto eval = evaluate_hand(cards);
    EXPECT_EQ(eval.type, HandType::FULL_HOUSE);
}

TEST(HandEvalTest, FourOfAKind) {
    // K♣ K♦ K♥ K♠ 2♣
    std::vector<Card> cards = {
        make_card(11, 0), // K♣
        make_card(11, 1), // K♦
        make_card(11, 2), // K♥
        make_card(11, 3), // K♠
        make_card(0, 0)   // 2♣
    };

    auto eval = evaluate_hand(cards);
    EXPECT_EQ(eval.type, HandType::FOUR_OF_A_KIND);
}

TEST(HandEvalTest, StraightFlush) {
    // 5♣ 6♣ 7♣ 8♣ 9♣
    std::vector<Card> cards = {
        make_card(3, 0),  // 5♣
        make_card(4, 0),  // 6♣
        make_card(5, 0),  // 7♣
        make_card(6, 0),  // 8♣
        make_card(7, 0)   // 9♣
    };

    auto eval = evaluate_hand(cards);
    EXPECT_EQ(eval.type, HandType::STRAIGHT_FLUSH);
}

TEST(HandEvalTest, RankSum) {
    // K♣ K♦ K♥ A♠ A♣ (Full House)
    // K=10, K=10, K=10, A=11, A=11 → sum=52
    std::vector<Card> cards = {
        make_card(11, 0), // K♣
        make_card(11, 1), // K♦
        make_card(11, 2), // K♥
        make_card(12, 3), // A♠
        make_card(12, 0)  // A♣
    };

    auto eval = evaluate_hand(cards);
    EXPECT_EQ(eval.rank_sum, 52);
}

TEST(ObservationTest, HasPair) {
    Hand hand = {
        make_card(11, 0), // K♣
        make_card(11, 1), // K♦
        make_card(0, 2),  // 2♥
        make_card(3, 3),  // 5♠
        make_card(7, 0),  // 9♣
        make_card(1, 1),  // 3♦
        make_card(5, 2),  // 7♥
        make_card(8, 3)   // 10♠
    };

    EXPECT_TRUE(has_pair(hand));
}

TEST(ObservationTest, HasThreeOfKind) {
    Hand hand = {
        make_card(11, 0), // K♣
        make_card(11, 1), // K♦
        make_card(11, 2), // K♥
        make_card(3, 3),  // 5♠
        make_card(7, 0),  // 9♣
        make_card(1, 1),  // 3♦
        make_card(5, 2),  // 7♥
        make_card(8, 3)   // 10♠
    };

    EXPECT_TRUE(has_three_of_kind(hand));
}

TEST(ObservationTest, FlushPotential) {
    Hand hand = {
        make_card(0, 0),  // 2♣
        make_card(3, 0),  // 5♣
        make_card(5, 0),  // 7♣
        make_card(7, 0),  // 9♣ (4 clubs)
        make_card(11, 1), // K♦
        make_card(1, 2),  // 3♥
        make_card(2, 3),  // 4♠
        make_card(8, 1)   // 10♦
    };

    EXPECT_TRUE(has_flush_potential(hand));
}

TEST(ObservationTest, MaxRankBucket) {
    Hand hand1 = {
        make_card(0, 0), make_card(1, 0), make_card(2, 0), make_card(3, 0),
        make_card(4, 0), make_card(5, 0), make_card(6, 0), make_card(7, 0)
    };
    EXPECT_EQ(get_max_rank_bucket(hand1), 2); // Max is 9 (rank 7) → bucket 2

    Hand hand2 = {
        make_card(0, 0), make_card(1, 0), make_card(2, 0), make_card(3, 0),
        make_card(4, 0), make_card(5, 0), make_card(6, 0), make_card(12, 0)
    };
    EXPECT_EQ(get_max_rank_bucket(hand2), 5); // Max is A (rank 12) → bucket 5
}
