#!/usr/bin/env python

import sys
import string
import logging
import configparser
import hyperparameters
from hyperparameters import *

import examples
import copy
import state
import stats
import diagnostics

def validate(cnt):
    import math
    logranks = []
    logging.info("BEGINNING VALIDATION AT TRAINING STEP %d" % cnt)
    logging.info(stats())
    i = 0
    for (i, ve) in enumerate(examples.get_validation_example()):
        logranks.append(math.log(m.validate(ve)))
        if (i+1) % 10 == 0:
            logging.info("Training step %d, validating example %d, mean(logrank) = %.2f, stddev(logrank) = %.2f", (cnt, i+1, numpy.mean(numpy.array(logranks)), numpy.std(numpy.array(logranks))))
            logging.info(stats())
    logging.info("FINAL VALIDATION AT TRAINING STEP %d: mean(logrank) = %.2f, stddev(logrank) = %.2f, cnt = %d", (cnt, numpy.mean(numpy.array(logranks)), numpy.std(numpy.array(logranks)), i+1))
    logging.info(stats())

if __name__ == "__main__":
    # import noise
    # indexed_weights = noise.indexed_weights()

    hyperparams = hyperparameters.load()

    import os.path, os

    logfile = os.path.join(hyperparams.get("data", "run_dir", "training.log")
    verboselogfile = os.path.join(RUN_DIR, "log")
    logging.basicConfig(filename=logfile, filemode="w", level=logging.DEBUG)
    logging.info("Logging to %s, and creating link %s" % (logfile, verboselogfile))

    import random, numpy
    random.seed(0)
    numpy.random.seed(0)

    import lexicon
    import model
    
    try:
        logging.info("Trying to read training state from %s..." % RUN_DIR)
        trainstate = state.load(RUN_DIR)
        logging.info("...success reading training state from %s" % RUN_DIR)
        logging.info("CONTINUING FROM TRAINING STATE")
    except IOError:
        logging.info("...FAILURE reading training state from %s" % RUN_DIR)
        logging.info("INITIALIZING")
        trainstate = state.TrainingState(RUN_DIR)
        logging.info("INITIALIZING TRAINING STATE")

    while True:
        trainstate.epoch()
