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
    hyperparameters = Hyperparameters("language-model.cfg")

    import os.path, os
    # Setting up a log file. This is handy to follow progress during
    # the program's execution without resorting to printing to stdout.
    logfile = os.path.join(hyperparameters.run_dir, hyperparameters.logfile)
    verboselogfile = os.path.join(hyperparameters.run_dir, hyperparameters.verboselogfile)
    logging.basicConfig(filename=logfile, filemode="w", level=logging.DEBUG)
    logging.info("Logging to %s, and creating link %s" % (logfile, verboselogfile))

    try:
        logging.info("Trying to read training state from %s..." % rundir)
        filename = os.path.join(hyperparameters.run_dir, "trainstate.pkl")
        with open(filename, 'rb') as f:
            saved_state = pickle.load(f)
            
        corpus_state, dictionary_state, hyperparameters = saved_state[0:2]
        
        corpus = Corpus(*corpus_state)
        dictionary = Dictionary(*dictionary_state)
        
        trainstate = TrainingState(corpus, dictionary, hyperparameters)
        trainstate.__setstate__(saved_state)
        logging.info("Successfully read training state from %s. Continuing from training state." % run_dir)
    except FileNotFoundError:
        logging.info("Failure reading training state from %s. Initialising a new model." % run_dir)

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
    for phase, size in enumerate(hyperparameters.curriculum_sizes):
        logging.info("Resizing dictionary ... ")
        trainstate.dictionary.enlarge(size)
        logging.info("Resized dictionary to size %s." % size)
        logging.info("Initialising curriculum phase %i." % phase)
        trainstate.epoch()
