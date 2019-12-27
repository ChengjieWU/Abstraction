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

indexer3rdRound = index.generalIndexer(3)


calculating_round = 2
indexer = index.generalIndexer(calculating_round)
cores = 220
chunck_num = 220


def calculating_feature_flop(hand):
    ret = np.zeros(8, dtype=np.float32)

    remaining_cards = cards.copy()
    for card in hand:
        remaining_cards.remove(card)
    
    count = len(remaining_cards)
    if count != 47:
        raise AssertionError( "remaining cards should be 47" )

    for left_public_card in remaining_cards:
        turn_board = hand + [left_public_card]
        turn_board_index = indexer3rdRound.index("".join(turn_board))
        ret += turn_hs_features[turn_board_index]

    ret /= count
    return ret


def calculating_hs_flop(param):
    save_result = True

    chunck_start = time.time()
    low = param[0]
    high = param[1]
    index_list = list(range(low, high))
    ret = np.zeros((len(index_list), 8), dtype=np.float32)
    for i, hand_index in enumerate(index_list):
        hand = indexer.canonicalHand(hand_index)
        hand = [hand[i:i+2] for i in range(0, len(hand), 2)]
        ret[i, :] = calculating_feature_flop(hand)
    
    if save_result:
        np.save("../data/flop_np/flop_{}_{}_{}.json".format(calculating_round, low, high), ret)
    
    chunck_end = time.time()
    logging.info("A chunck finished with {} time consumed".format(chunck_end - chunck_start))
    return ret


if __name__ == "__main__":
    start_time = time.time()
    total_jobs = indexer.getSize(calculating_round)

    logging.info("Loading turn HS features...")
    turn_hs_features = np.load("../data/turn.npy")

    jobs_list = list()
    chunck_size = total_jobs // chunck_num
    for i in range(chunck_num - 1):
        jobs_list.append((i * chunck_size, (i+1) * chunck_size))
    jobs_list.append(((chunck_num - 1) * chunck_size, total_jobs))
    logging.debug("Jobs are: " + str(jobs_list))

    logging.info("Round {}, {} cores, {} chuncks, calculating...".format(calculating_round, cores, chunck_num))
    pool = multiprocessing.Pool(cores)
    hand_lists = pool.map(calculating_hs_flop, jobs_list)
    pool.close()
    pool.join()

    logging.info("Collecting results...")
    hand_lists = np.array(hand_lists)
    hand_lists = np.concatenate(hand_lists, 0)

    logging.info("Saving numpy file...")
    np.save("../data/flop.npy", hand_lists)

    logging.info("Done.")

    end_time = time.time()
    logging.info("Time consumed: {}".format(end_time - start_time))