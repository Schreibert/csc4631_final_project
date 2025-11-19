#include <gtest/gtest.h>
#include "balatro/card.hpp"

using namespace balatro;

TEST(CardTest, CardEncoding) {
    // Test rank and suit extraction
    Card c = make_card(12, 3); // Ace of Spades
    EXPECT_EQ(get_rank(c), 12);
    EXPECT_EQ(get_suit(c), 3);

    Card c2 = make_card(0, 0); // 2 of Clubs
    EXPECT_EQ(get_rank(c2), 0);
    EXPECT_EQ(get_suit(c2), 0);
}

TEST(CardTest, RankValues) {
    // Test rank value mapping for scoring
    EXPECT_EQ(get_rank_value(0), 2);   // 2
    EXPECT_EQ(get_rank_value(8), 10);  // 10
    EXPECT_EQ(get_rank_value(9), 10);  // J
    EXPECT_EQ(get_rank_value(10), 10); // Q
    EXPECT_EQ(get_rank_value(11), 10); // K
    EXPECT_EQ(get_rank_value(12), 11); // A
}

TEST(DeckTest, Initialization) {
    Deck deck;
    deck.reset(42);

    EXPECT_EQ(deck.remaining(), DECK_SIZE);
    EXPECT_FALSE(deck.is_empty());
}

TEST(DeckTest, DeterministicShuffle) {
    Deck deck1, deck2;
    deck1.reset(12345);
    deck2.reset(12345);

    // Same seed should produce same sequence
    for (int i = 0; i < DECK_SIZE; ++i) {
        EXPECT_EQ(deck1.draw(), deck2.draw());
    }
}

TEST(DeckTest, Drawing) {
    Deck deck;
    deck.reset(100);

    // Draw all 52 cards
    std::vector<Card> drawn;
    for (int i = 0; i < DECK_SIZE; ++i) {
        Card c = deck.draw();
        EXPECT_NE(c, 255);
        drawn.push_back(c);
    }

    // Check all cards are unique
    std::sort(drawn.begin(), drawn.end());
    for (int i = 0; i < DECK_SIZE; ++i) {
        EXPECT_EQ(drawn[i], i);
    }
}

TEST(DeckTest, Reshuffle) {
    Deck deck;
    deck.reset(200);

    // Draw some cards and discard them
    for (int i = 0; i < 10; ++i) {
        Card c = deck.draw();
        deck.discard(c);
    }

    // Draw remaining cards to trigger reshuffle
    for (int i = 0; i < DECK_SIZE - 10; ++i) {
        deck.draw();
    }

    // Now drawing should trigger reshuffle of discards
    Card c = deck.draw();
    EXPECT_NE(c, 255); // Should successfully draw after reshuffle
}

TEST(HandTest, DealHand) {
    Deck deck;
    deck.reset(300);
    Hand hand;

    deal_hand(deck, hand);

    // Should have 8 cards
    for (int i = 0; i < HAND_SIZE; ++i) {
        EXPECT_LT(hand[i], DECK_SIZE);
    }

    EXPECT_EQ(deck.remaining(), DECK_SIZE - HAND_SIZE);
}

TEST(HandTest, ReplaceCards) {
    Deck deck;
    deck.reset(400);
    Hand hand;

    deal_hand(deck, hand);
    Card old_card = hand[0];

    // Replace first two cards
    replace_cards(deck, hand, {0, 1});

    // First card should be different (very high probability)
    EXPECT_NE(hand[0], old_card);

    // Should have drawn 2 more cards
    EXPECT_EQ(deck.remaining(), DECK_SIZE - HAND_SIZE - 2);
}
