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

indexer4thRound = index.generalIndexer(4)


distribution_aware = True
calculating_round = 3
indexer = index.generalIndexer(calculating_round)
cores = 220
chunck_num = 2000


def calculating_feature_turn(hand):
    remaining_cards = cards.copy()
    for card in hand:
        remaining_cards.remove(card)
    
    count = len(remaining_cards)
    if count != 46:
        raise AssertionError( "remaining cards should be 46" )

    if not distribution_aware:
        ret = np.zeros(8, dtype=np.float32)
        for left_public_card in remaining_cards:
            river_board = hand + [left_public_card]
            river_board_index = indexer4thRound.index("".join(river_board))
            ret += river_hs_features[river_board_index]
        ret /= count
    else:
        ret = np.zeros(20, dtype=np.float32)
        for left_public_card in remaining_cards:
            river_board = hand + [left_public_card]
            river_board_index = indexer4thRound.index("".join(river_board))
            hs = np.average(river_hs_features[river_board_index])
            hs_slot = min(int(hs * 20), 19)
            ret[hs_slot] += 1
        if np.sum(ret) != count:
            raise AssertionError( "sum of probability mass should be 1" )

        ret /= count

    return ret


def calculating_hs_turn(param):
    save_result = True

    chunck_start = time.time()
    low = param[0]
    high = param[1]
    index_list = list(range(low, high))
    if not distribution_aware:
        ret = np.zeros((len(index_list), 8), dtype=np.float32)
    else:
        ret = np.zeros((len(index_list), 20), dtype=np.float32)
    for i, hand_index in enumerate(index_list):
        hand = indexer.canonicalHand(hand_index)
        hand = [hand[i:i+2] for i in range(0, len(hand), 2)]
        ret[i, :] = calculating_feature_turn(hand)
    
    if save_result:
        if not distribution_aware:
            np.save("../data/turn_np/turn_{}_{}_{}.npy".format(calculating_round, low, high), ret)
        else:
            np.save("../data/distribution_turn_np/turn_{}_{}_{}.npy".format(calculating_round, low, high), ret)
    
    chunck_end = time.time()
    logging.info("A chunck finished with {} time consumed".format(chunck_end - chunck_start))
    return ret


if __name__ == "__main__":
    start_time = time.time()
    total_jobs = indexer.getSize(calculating_round)

    logging.info("Loading river HS features...")
    river_hs_features = np.load("../data/river.npy")

    jobs_list = list()
    chunck_size = total_jobs // chunck_num
    for i in range(chunck_num - 1):
        jobs_list.append((i * chunck_size, (i+1) * chunck_size))
    jobs_list.append(((chunck_num - 1) * chunck_size, total_jobs))
    logging.debug("Jobs are: " + str(jobs_list))

    logging.info("Round {}, {} cores, {} chuncks, calculating...".format(calculating_round, cores, chunck_num))
    pool = multiprocessing.Pool(cores)
    hand_lists = pool.map(calculating_hs_turn, jobs_list)
    pool.close()
    pool.join()

    logging.info("Collecting results...")
    hand_lists = np.array(hand_lists)
    hand_lists = np.concatenate(hand_lists, 0)

    logging.info("Saving numpy file...")
    if not distribution_aware:
        np.save("../data/turn.npy", hand_lists)
    else:
        np.save("../data/distribution_turn.npy", hand_lists)

    logging.info("Done.")

    end_time = time.time()
    logging.info("Time consumed: {}".format(end_time - start_time))