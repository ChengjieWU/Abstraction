import multiprocessing
import pickle
import logging
import json
import os

import numpy as np

import index


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)


def calculate_canonical_hands(param):
    low = param[0]
    high = param[1]
    r = param[2]
    # indexer_in = index.generalIndexer(r)

    index_list = list(range(low, high))
    hand_list = list(map(indexer.canonicalHand, index_list))
    
    return hand_list



calculating_round = 4
cores = os.cpu_count()
chunck_num = 2000
indexer = index.generalIndexer(calculating_round)


if __name__ == "__main__":
    total_jobs = indexer.getSize(calculating_round)

    jobs_list = list()
    chunck_size = total_jobs // chunck_num
    for i in range(chunck_num - 1):
        jobs_list.append((i * chunck_size, (i+1) * chunck_size, calculating_round))
    jobs_list.append(((chunck_num - 1) * chunck_size, total_jobs, calculating_round))
    print(jobs_list)

    logging.info("Round {}, {} cores, {} chuncks, calculating...".format(calculating_round, cores, chunck_num))
    pool = multiprocessing.Pool(cores)
    # hand_lists = pool.map(calculate_canonical_hands, jobs_list, chunksize=10)
    hand_lists = pool.map(calculate_canonical_hands, jobs_list)
    pool.close()
    pool.join()
    logging.info("Collecting results...")
    hands = list()
    for l in hand_lists:
        hands += l
    
    logging.info("Finished generating, {} canonical hands in total.".format(len(hands)))
    
    logging.info("Saving pickle file...")
    with open("./canonicalHands_{}.pkl".format(calculating_round), "wb") as fp:
        pickle.dump(hands, fp)
    
    logging.info("Saving json file...")
    hands_dict = {calculating_round: hands}
    with open("./canonicalHands_{}.json".format(calculating_round), "w") as fp:
        json.dump(hands_dict, fp)

    logging.info("Done.")
