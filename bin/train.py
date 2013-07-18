#!/usr/bin/env python

import logging
import hyperparameters

import examples
import state
import stats
from lexicon import corpus, dictionary 

def validate(cnt):
    import math
    logranks = []
    logging.info("BEGINNING VALIDATION AT TRAINING STEP %d" % cnt)
    logging.info(stats())
    i = 0
    for (i, ve) in enumerate(examples.get_validation_example()):
        logranks.append(math.log(trainstate.model.validate(ve)))
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

    run_dir = hyperparams.get("data", "run_dir")

    logfile = os.path.join(hyperparams.get("data", "run_dir"), hyperparams.get("training", "logfile"))
    verboselogfile = os.path.join(run_dir, hyperparams.get("training", "verboselogfile"))
    logging.basicConfig(filename=logfile, filemode="w", level=logging.DEBUG)
    logging.info("Logging to %s, and creating link %s" % (logfile, verboselogfile))

    import random, numpy
    random.seed(0)
    numpy.random.seed(0)
    
    try:
        logging.info("Trying to read training state from %s..." % run_dir)
        trainstate = state.load(run_dir)
        logging.info("...success reading training state from %s" % run_dir)
        logging.info("CONTINUING FROM TRAINING STATE")
    except IOError:
        logging.info("...FAILURE reading training state from %s" % run_dir)
        logging.info("INITIALIZING")
        training_sentences = hyperparams.get("data", "training_sentences")
        training_corpus = Corpus(training_sentences)
        vocab_size = hyperparams.get("training", "vocab_size")
        dictionary = Dictionary(training_corpus, vocab_size)
        window_size = hyperparams.get("training", "window_size")
        batch_size = hyperparams.get("training", "batch_size")
        validate_every = hyperparams.get("training", "validate_every")
        embedding_size = hyperparams.get("training", "embedding_size")
        trainstate = state.TrainingState(rundir = run_dir, corpus = training_corpus, dictionary = dictionary, window_size = window_size, batch_size = batch_size, validate_every = validate_every)
        logging.info("INITIALIZING TRAINING STATE")

    while True:
        trainstate.epoch()
