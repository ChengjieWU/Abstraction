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


def making_numpy_data(param):
    save_result = False

    chunck_start = time.time()
    low = param[0]
    high = param[1]
    index_list = list(range(low, high))

    with open("../data/features/features_{}_{}_{}.json".format(calculating_round, low, high), "r") as fp:
        a = json.load(fp)
    b = np.array(a["data"], dtype=np.float32)
    
    if save_result:
        np.save("../data/river_np/river_{}_{}_{}.npy".format(calculating_round, low, high), b)
    
    chunck_end = time.time()
    logging.info("A chunck finished with {} time consumed".format(chunck_end - chunck_start))
    
    return b


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
    hand_lists = pool.map(making_numpy_data, jobs_list)
    pool.close()
    pool.join()

    logging.info("Collecting results...")
    hand_lists = np.array(hand_lists)
    hand_lists = np.concatenate(hand_lists, 0)

    logging.info("Saving numpy file...")
    np.save("../data/river.npy", hand_lists)

    logging.info("Done.")

    end_time = time.time()
    logging.info("Time consumed: {}".format(end_time - start_time))