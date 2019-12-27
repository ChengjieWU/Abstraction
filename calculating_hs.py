import itertools
import random
import multiprocessing
import pickle
import logging
import json
import os
import time

import numpy as np

import handrank
import index


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)


ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
suits = ["s", "h", "c", "d"]
cards = [a + b for a in ranks for b in suits]

num_ranks = len(ranks)
num_suits = len(suits)
num_cards = len(cards)

canonical2domain = {
    "2s2h": 4, 
    "2s3h": 1, "3s3h": 6, 
    "4s2h": 1, "4s3h": 1, "4s4h": 6, 
    "5s2h": 1, "5s3h": 1, "5s4h": 1, "5s5h": 6, 
    "6s2h": 1, "6s3h": 1, "6s4h": 1, "6s5h": 1, "6s6h": 7, 
    "7s2h": 1, "7s3h": 1, "7s4h": 1, "7s5h": 2, "7s6h": 3, "7s7h": 7, 
    "8s2h": 1, "8s3h": 1, "8s4h": 2, "8s5h": 2, "8s6h": 3, "8s7h": 3, "8s8h": 8, 
    "9s2h": 2, "9s3h": 2, "9s4h": 2, "9s5h": 2, "9s6h": 3, "9s7h": 3, "9s8h": 3, "9s9h": 8, 
    "Ts2h": 2, "Ts3h": 2, "Ts4h": 2, "Ts5h": 2, "Ts6h": 3, "Ts7h": 3, "Ts8h": 3, "Ts9h": 5, "TsTh": 8, 
    "Js2h": 2, "Js3h": 2, "Js4h": 4, "Js5h": 4, "Js6h": 4, "Js7h": 4, "Js8h": 5, "Js9h": 5, "JsTh": 5, "JsJh": 8, 
    "Qs2h": 4, "Qs3h": 4, "Qs4h": 4, "Qs5h": 4, "Qs6h": 4, "Qs7h": 4, "Qs8h": 5, "Qs9h": 5, "QsTh": 5, "QsJh": 5, "QsQh": 8, 
    "Ks2h": 4, "Ks3h": 4, "Ks4h": 4, "Ks5h": 6, "Ks6h": 6, "Ks7h": 6, "Ks8h": 6, "Ks9h": 6, "KsTh": 7, "KsJh": 7, "KsQh": 7, "KsKh": 8, 
    "As2h": 6, "As3h": 6, "As4h": 6, "As5h": 6, "As6h": 6, "As7h": 6, "As8h": 6, "As9h": 7, "AsTh": 7, "AsJh": 7, "AsQh": 7, "AsKh": 7, "AsAh": 8, 
    "2s3s": 1, 
    "2s4s": 1, "3s4s": 1, 
    "2s5s": 1, "3s5s": 1, "4s5s": 1, 
    "2s6s": 1, "3s6s": 1, "4s6s": 1, "5s6s": 3, 
    "2s7s": 1, "3s7s": 1, "4s7s": 2, "5s7s": 3, "6s7s": 3, 
    "2s8s": 2, "3s8s": 2, "4s8s": 2, "5s8s": 3, "6s8s": 3, "7s8s": 3, 
    "2s9s": 2, "3s9s": 2, "4s9s": 2, "5s9s": 3, "6s9s": 3, "7s9s": 3, "8s9s": 3, 
    "2sTs": 2, "3sTs": 3, "4sTs": 3, "5sTs": 3, "6sTs": 3, "7sTs": 5, "8sTs": 5, "9sTs": 5, 
    "2sJs": 4, "3sJs": 4, "4sJs": 4, "5sJs": 4, "6sJs": 4, "7sJs": 5, "8sJs": 5, "9sJs": 5, "TsJs": 5, 
    "2sQs": 4, "3sQs": 4, "4sQs": 4, "5sQs": 4, "6sQs": 5, "7sQs": 5, "8sQs": 5, "9sQs": 5, "TsQs": 7, "JsQs": 7, 
    "2sKs": 4, "3sKs": 6, "4sKs": 6, "5sKs": 6, "6sKs": 6, "7sKs": 6, "8sKs": 6, "9sKs": 7, "TsKs": 7, "JsKs": 7, "QsKs": 7, 
    "2sAs": 6, "3sAs": 6, "4sAs": 6, "5sAs": 6, "6sAs": 6, "7sAs": 7, "8sAs": 7, "9sAs": 7, "TsAs": 7, "JsAs": 7, "QsAs": 7, "KsAs": 7
}
indexer1stRound = index.generalIndexer(1)
size1st = indexer1stRound.getSize(1)
index2domain = [canonical2domain[indexer1stRound.canonicalHand(i)] for i in range(size1st)]


def calculating_equity(c1, c2):
    c1 = "".join(c1)
    c2 = "".join(c2)
    r1 = handrank.rankHand(c1)
    r2 = handrank.rankHand(c2)
    if (r1 > r2):
        return 1
    elif (r1 == r2):
        return 0.5
    else:
        return 0


def hs_sampling(hand, iter_num=1000):
    remaining_cards = cards.copy()
    for card in hand:
        remaining_cards.remove(card)
    bhs = np.zeros(20)

    for _ in range(iter_num):
        public_board = tuple(random.sample(remaining_cards, 5))
        opponent_possible_cards = remaining_cards[:]
        for card in public_board:
            opponent_possible_cards.remove(card)
        hs = 0
        for opponent_hand in itertools.combinations(opponent_possible_cards, 2):
            hs += calculating_equity(hand + public_board, opponent_hand + public_board)
        hs /= len(opponent_possible_cards) * (len(opponent_possible_cards) - 1) / 2
        bhs[int(hs * 20) if hs < 1 else 19] += 1
        
    bhs /= iter_num
    return (hand, bhs)


def calculating_feature(hand):
    ret = np.zeros(8, dtype=np.float32)
    count = np.zeros(8, dtype=np.float32)

    public_board = tuple(hand[2:])
    remaining_cards = cards.copy()
    for card in hand:
        remaining_cards.remove(card)

    for opponent_hand in itertools.combinations(remaining_cards, 2):
        feature_index = index2domain[indexer1stRound.index("".join(opponent_hand))] - 1
        hs = calculating_equity(hand, opponent_hand + public_board)
        ret[feature_index] += hs
        count[feature_index] += 1

    ret = ret / count
    return ret


def calculating_hs(param):
    save_result = True

    chunck_start = time.time()
    low = param[0]
    high = param[1]
    index_list = list(range(low, high))
    ret = np.zeros((len(index_list), 8), dtype=np.float32)
    for i, hand_index in enumerate(index_list):
        hand = indexer.canonicalHand(hand_index)
        hand = [hand[i:i+2] for i in range(0, len(hand), 2)]
        ret[i, :] = calculating_feature(hand)
    
    if save_result:
        data = ret.tolist()
        hands_dict = {"low": low, "high": high, "data": data}
        with open("./features/features_{}_{}_{}.json".format(calculating_round, low, high), "w") as fp:
            json.dump(hands_dict, fp)
    
    chunck_end = time.time()
    logging.info("A chunck finished with {} time consumed".format(chunck_end - chunck_start))
    return ret


calculating_round = 4
indexer = index.generalIndexer(calculating_round)
cores = 220
chunck_num = 200000

if __name__ == "__main__":
    start_time = time.time()
    total_jobs = indexer.getSize(calculating_round)

    jobs_list = list()
    chunck_size = total_jobs // chunck_num
    for i in range(chunck_num - 1):
        jobs_list.append((i * chunck_size, (i+1) * chunck_size))
    jobs_list.append(((chunck_num - 1) * chunck_size, total_jobs))
    logging.debug("Jobs are: " + str(jobs_list))

    logging.info("Round {}, {} cores, {} chuncks, calculating...".format(calculating_round, cores, chunck_num))
    pool = multiprocessing.Pool(cores)
    hand_lists = pool.map(calculating_hs, jobs_list)
    pool.close()
    pool.join()

    logging.info("Collecting results...")
    hand_lists = np.array(hand_lists)
    hand_lists = np.concatenate(hand_lists, 0)

    logging.info("Saving json file...")
    hand_lists = hand_lists.tolist()
    hands_dict = {calculating_round: hand_lists}
    with open("./features_{}.json".format(calculating_round), "w") as fp:
        json.dump(hands_dict, fp)

    logging.info("Done.")

    end_time = time.time()
    logging.info("Time consumed: {}".format(end_time - start_time))
