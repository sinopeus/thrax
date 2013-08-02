#!/usr/bin/env python

import logging, pickle
from hyperparameters import Hyperparameters


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
        logging.info("Trying to read training state from %s..." % hyperparameters.run_dir)
        filename = os.path.join(hyperparameters.run_dir, "trainstate.pkl")
        with open(filename, 'rb') as f:
            saved_state = pickle.load(f)

        corpus_state, dictionary_state, hyperparameters = saved_state[0:2]

        from lexicon import Corpus, Dictionary
        corpus = Corpus(*corpus_state)
        dictionary = Dictionary(*dictionary_state)

        from state import TrainingState
        trainstate = TrainingState(corpus, dictionary, hyperparameters)
        trainstate.__setstate__(saved_state)
        logging.info("Successfully read training state from %s. Training may begin." % hyperparameters.run_dir)
    except FileNotFoundError:
        logging.info("Failure reading training state from %s. Initialising a new state." % hyperparameters.run_dir)

        from lexicon import Corpus, Dictionary
        logging.info("Processing training corpus ...")
        training_corpus = Corpus(os.path.join(hyperparameters.data_dir, hyperparameters.training_sentences))
        hyperparameters.vocab_size = training_corpus.lexicon_size()
        logging.info("Training corpus processed, initialising dictionary ...")
        dictionary = Dictionary(training_corpus, hyperparameters.curriculum_sizes[0])
        logging.info("Dictionary initialised, proceeding with training.")

        from state import TrainingState
        trainstate = TrainingState(training_corpus, dictionary, hyperparameters)
        logging.info("State initialised.")

    from validate import Validator
    logging.info("Processing validation corpus ...")
    validation_corpus = Corpus(os.path.join(hyperparameters.data_dir, hyperparameters.validation_sentences))
    logging.info("Validation corpus processed. Initialising model validator.")
    validator = Validator(validation_corpus, trainstate.model)
    logging.info("Model validator initialised.")

    input("Press any key to continue...")
    for phase, size in enumerate(hyperparameters.curriculum_sizes):
        logging.info("Resizing dictionary ... ")
        trainstate.dictionary.enlarge(size)
        logging.info("Resized dictionary to size %s." % size)
        logging.info("Initialising curriculum phase %i." % phase)
        trainstate.epoch()
