#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#define __STDC_LIMIT_MACROS

#include <stdint.h>
#include <bitset>
#include <sstream>

#include "evalHandTables"

#include "pybind11/pybind11.h"
namespace py = pybind11;


#define MAX_BOARD_CARDS 5
#define MAX_HOLE_CARDS 2
#define MAX_SUITS 4
#define MAX_RANKS 13

#define rankOfCard(card) ((card)/MAX_SUITS)
#define suitOfCard(card) ((card)%MAX_SUITS)
#define makeCard(rank, suit) ((rank)*MAX_SUITS+(suit))


constexpr int kMaxSuits = 4;

std::string kSuitChars = "cdhs";
std::string kRankChars = "23456789TJQKA";

using I = uint64_t;

auto dump(I v) { return std::bitset<sizeof(I) * __CHAR_BIT__>(v); }

I bit_twiddle_permute(I v) {
    I t = v | (v - 1);
    I w = (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctzl(v) + 1));
    return w;
}

// This is an equivalent wrapper to acpc evalHandTables.Cardset.
// It stores the cards for each color over 16 * 4 bits. The use of a Union
// allows to access only a specific color (16 bits) using bySuit[color].
// A uint8_t card is defined by the integer <rank> * MAX_SUITS + <suit>
class CardSet {
public:
    union CardSetUnion {
        CardSetUnion() : cards(0) {}

        uint16_t bySuit[kMaxSuits];
        uint64_t cards;
    } cs;

public:
    CardSet() : cs() {}

    explicit CardSet(std::string cardString);

    explicit CardSet(std::vector<int> cards);

    // Returns a set containing num_ranks cards per suit for num_suits.
    CardSet(uint16_t num_suits, uint16_t num_ranks);

    std::string ToString() const;

    // Returns the cards present in this set in ascending order.
    std::vector<uint8_t> ToCardArray() const;

    // Add a card, as MAX_RANKS * <suite> + <rank> to the CardSet.
    void AddCard(uint8_t card);

    // Toogle (does not remove) the bit associated to `card`.
    void RemoveCard(uint8_t card);

    bool ContainsCards(uint8_t card) const;

    int NumCards() const;

    // Returns the ranking value of this set of cards as evaluated by ACPC.
    int RankCards() const;

    // TODO(author2): Remove?
    std::vector<CardSet> SampleCards(int nbCards);
};


CardSet::CardSet(std::string cardString) : cs() {
    for (int i = 0; i < cardString.size(); i += 2) {
        char rankChr = cardString[i];
        char suitChr = cardString[i + 1];

        uint8_t rank = (uint8_t) (kRankChars.find(rankChr));
        uint8_t suit = (uint8_t) (kSuitChars.find(suitChr));

        cs.bySuit[suit] |= ((uint16_t) 1 << rank);
    }
}

CardSet::CardSet(std::vector<int> cards) : cs() {
    for (int i = 0; i < cards.size(); ++i) {
        int rank = rankOfCard(cards[i]);
        int suit = suitOfCard(cards[i]);

        cs.bySuit[suit] |= ((uint16_t) 1 << rank);
    }
}

CardSet::CardSet(uint16_t num_suits, uint16_t num_ranks) : cs() {
    for (uint16_t r = 0; r < num_ranks; r++) {
        for (uint16_t s = 0; s < num_suits; s++) {
            cs.bySuit[s] |= ((uint16_t) 1 << r);
        }
    }
}

std::string CardSet::ToString() const {
    std::ostringstream result;
    for (int r = MAX_RANKS - 1; r >= 0; r--) {
        for (int s = MAX_SUITS - 1; s >= 0; s--) {
            uint32_t mask = (uint32_t) 1 << r;
            if (cs.bySuit[s] & mask) {
                result << kRankChars[r] << kSuitChars[s];
            }
        }
    }

    return result.str();
}

std::vector<uint8_t> CardSet::ToCardArray() const {
    std::vector<uint8_t> result(NumCards(), 0);

    int i = 0;
    for (int r = 0; r < MAX_RANKS; ++r) {
        for (int s = 0; s < MAX_SUITS; ++s) {
            uint32_t mask = (uint32_t) 1 << r;
            if (cs.bySuit[s] & mask) {
                result[i++] = makeCard(r, s);
            }
        }
    }
    return result;
}

void CardSet::AddCard(uint8_t card) {
    int rank = rankOfCard(card);
    int suit = suitOfCard(card);

    cs.bySuit[suit] |= ((uint16_t) 1 << rank);
}

void CardSet::RemoveCard(uint8_t card) {
    int rank = rankOfCard(card);
    int suit = suitOfCard(card);

    cs.bySuit[suit] ^= ((uint16_t) 1 << rank);
}

int CardSet::NumCards() const { return __builtin_popcountl(cs.cards); }

int CardSet::RankCards() const {
    ::Cardset csNative;
    csNative.cards = cs.cards;
    return rankCardset(csNative);
}

std::vector<CardSet> CardSet::SampleCards(int nbCards) {
    std::vector<CardSet> combinations;

    uint64_t p = 0;
    for (int i = 0; i < nbCards; ++i) {
        p += (1 << i);
    }

    for (I n = bit_twiddle_permute(p);
         n > p; p = n, n = bit_twiddle_permute(p)) {
        uint64_t combo = n & cs.cards;
        if (__builtin_popcountl(combo) == nbCards) {
            CardSet c;
            c.cs.cards = combo;
            combinations.emplace_back(c);
        }
    }

    // std::cout << "combinations.size() " << combinations.size() << std::endl;
    return combinations;
}

bool CardSet::ContainsCards(uint8_t card) const {
    int rank = rankOfCard(card);
    int suit = suitOfCard(card);
    return (cs.bySuit[suit] & ((uint16_t) 1 << rank)) > 0;
}

int rankHand(std::string str) {
    CardSet a(str);
    return a.RankCards();
}

std::string printHand(std::string str) {
    CardSet a(str);
    return a.ToString();
}

PYBIND11_MODULE(handrank, m) {
    m.doc() = "Calculating Hand Rank"; // optional module docstring
    m.def("rankHand", &rankHand, "Calculate hand rank");
    m.def("printHand", &printHand, "Print hand");
}
