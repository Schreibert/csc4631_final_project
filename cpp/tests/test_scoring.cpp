#include <gtest/gtest.h>
#include "balatro/scoring.hpp"

using namespace balatro;

TEST(ScoringTest, HighCard) {
    // Single card: A♠
    std::vector<Card> cards = {make_card(12, 3)}; // A
    auto eval = evaluate_hand(cards);
    int score = calculate_score(eval);

    // base_chips=5, rank_sum=11, base_mult=1
    // (5 + 11) × 1 = 16
    EXPECT_EQ(score, 16);
}

TEST(ScoringTest, Pair) {
    // K♣ K♦
    std::vector<Card> cards = {
        make_card(11, 0), // K
        make_card(11, 1)  // K
    };
    auto eval = evaluate_hand(cards);
    int score = calculate_score(eval);

    // base_chips=10, rank_sum=20 (10+10), base_mult=2
    // (10 + 20) × 2 = 60
    EXPECT_EQ(score, 60);
}

TEST(ScoringTest, FullHouseGolden) {
    // K♣ K♦ K♥ A♠ A♣ (the example from the docs)
    std::vector<Card> cards = {
        make_card(11, 0), // K
        make_card(11, 1), // K
        make_card(11, 2), // K
        make_card(12, 3), // A
        make_card(12, 0)  // A
    };
    auto eval = evaluate_hand(cards);
    int score = calculate_score(eval);

    // base_chips=40, rank_sum=52 (10+10+10+11+11), base_mult=4
    // (40 + 52) × 4 = 368
    EXPECT_EQ(score, 368);
}

TEST(ScoringTest, Straight) {
    // 5♣ 6♦ 7♥ 8♠ 9♣
    std::vector<Card> cards = {
        make_card(3, 0),  // 5
        make_card(4, 1),  // 6
        make_card(5, 2),  // 7
        make_card(6, 3),  // 8
        make_card(7, 0)   // 9
    };
    auto eval = evaluate_hand(cards);
    int score = calculate_score(eval);

    // base_chips=30, rank_sum=35 (5+6+7+8+9), base_mult=4
    // (30 + 35) × 4 = 260
    EXPECT_EQ(score, 260);
}

TEST(ScoringTest, Flush) {
    // 2♣ 5♣ 7♣ 9♣ K♣
    std::vector<Card> cards = {
        make_card(0, 0),  // 2
        make_card(3, 0),  // 5
        make_card(5, 0),  // 7
        make_card(7, 0),  // 9
        make_card(11, 0)  // K
    };
    auto eval = evaluate_hand(cards);
    int score = calculate_score(eval);

    // base_chips=35, rank_sum=33 (2+5+7+9+10), base_mult=4
    // (35 + 33) × 4 = 272
    EXPECT_EQ(score, 272);
}

TEST(ScoringTest, FourOfAKind) {
    // A♣ A♦ A♥ A♠ K♣
    std::vector<Card> cards = {
        make_card(12, 0), // A
        make_card(12, 1), // A
        make_card(12, 2), // A
        make_card(12, 3), // A
        make_card(11, 0)  // K
    };
    auto eval = evaluate_hand(cards);
    int score = calculate_score(eval);

    // base_chips=60, rank_sum=54 (11+11+11+11+10), base_mult=7
    // (60 + 54) × 7 = 798
    EXPECT_EQ(score, 798);
}

TEST(ScoringTest, StraightFlush) {
    // 9♠ 10♠ J♠ Q♠ K♠
    std::vector<Card> cards = {
        make_card(7, 3),  // 9
        make_card(8, 3),  // 10
        make_card(9, 3),  // J
        make_card(10, 3), // Q
        make_card(11, 3)  // K
    };
    auto eval = evaluate_hand(cards);
    int score = calculate_score(eval);

    // base_chips=100, rank_sum=49 (9+10+10+10+10), base_mult=8
    // (100 + 49) × 8 = 1192
    EXPECT_EQ(score, 1192);
}

TEST(ScoringTest, ThreeOfAKind) {
    // 5♣ 5♦ 5♥
    std::vector<Card> cards = {
        make_card(3, 0), // 5
        make_card(3, 1), // 5
        make_card(3, 2)  // 5
    };
    auto eval = evaluate_hand(cards);
    int score = calculate_score(eval);

    // base_chips=30, rank_sum=15 (5+5+5), base_mult=3
    // (30 + 15) × 3 = 135
    EXPECT_EQ(score, 135);
}

TEST(ScoringTest, TwoPair) {
    // 9♣ 9♦ 5♥ 5♠
    std::vector<Card> cards = {
        make_card(7, 0), // 9
        make_card(7, 1), // 9
        make_card(3, 2), // 5
        make_card(3, 3)  // 5
    };
    auto eval = evaluate_hand(cards);
    int score = calculate_score(eval);

    // base_chips=20, rank_sum=28 (9+9+5+5), base_mult=2
    // (20 + 28) × 2 = 96
    EXPECT_EQ(score, 96);
}
