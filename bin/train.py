#!/usr/bin/env python

import logging, state, stats, examples
from hyperparameters import Hyperparameters

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

    hyperparameters = Hyperparameters("language-model.cfg")

    import os.path, os

    run_dir = hyperparameters.run_dir

    logfile = os.path.join(hyperparameters.run_dir, hyperparameters.logfile)
    verboselogfile = os.path.join(run_dir, hyperparameters.verboselogfile)
    logging.basicConfig(filename=logfile, filemode="w", level=logging.DEBUG)
    logging.info("Logging to %s, and creating link %s" % (logfile, verboselogfile))

    import random, numpy
    random.seed(0)
    numpy.random.seed(0)

    try:
        trainstate = state.load(run_dir)
        logging.info("...success reading training state from %s" % run_dir)
        logging.info("CONTINUING FROM TRAINING STATE")
    except FileNotFoundError:
        logging.info("...FAILURE reading training state from %s" % run_dir)
        logging.info("INITIALIZING")

        from lexicon import Corpus, Dictionary
        from state import TrainingState

        logging.info("Processing corpus ...")
        training_corpus = Corpus(os.path.join(hyperparameters.data_dir, hyperparameters.training_sentences))
        hyperparameters.vocab_size = training_corpus.lexicon_size()
        logging.info("Corpus processed, initialising dictionary ...")
        dictionary = Dictionary(training_corpus, hyperparameters.curriculum_sizes[0])
        logging.info("Dictionary initialised, proceeding with training.")
        trainstate = TrainingState(training_corpus, dictionary, hyperparameters)

    input("Press Enter to continue...")
    for size in hyperparameters.curriculum_sizes:
        logging.info("Resizing dictionary ... ")
        trainstate.dictionary.rebuild(size)
        logging.info("Resized dictionary to size %s." % size)
        trainstate.epoch()
