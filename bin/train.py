#!/usr/bin/env python

import sys
import string
import logging
import configparser
import hyperparameters

import examples
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

    runtimeconfig = copy.deepcopy(hyperparameters.config)
    rundir = runtimeconfig.get("data", "run_dir")
    # common.dump.create_canonical_directory(HYPERPARAMETERS)

    import os.path, os
    logfile = os.path.join(rundir, "training.log")
    if newkeystr != "":
        verboselogfile = os.path.join(rundir, "log%s" % newkeystr)
        logging.info("Logging to %s, and creating link %s" % (logfile, verboselogfile))
        os.system("ln -s log %s " % (verboselogfile))
    else:
        print("Logging to %s, not creating any link because of default settings" % logfile, file=sys.stderr)

    import random, numpy
    random.seed(miscglobals.RANDOMSEED)
    numpy.random.seed(miscglobals.RANDOMSEED)

    import lexicon
    import model

    logging.basicConfig(filename=logfile, filemode="w", level=logging.DEBUG)
    
    try:
        logging.info("Trying to read training state for %s %s..." % (newkeystr, rundir))
        (m, cnt, epoch, get_train_minibatch) = state.load(rundir, newkeystr)
        logging.info("...success reading training state for %s %s" % (newkeystr, rundir))
        logging.info("CONTINUING FROM TRAINING STATE")
    except IOError:
        logging.info("...FAILURE reading training state for %s %s" % (newkeystr, rundir))
        logging.info("INITIALIZING")

        m = model.Model()
        cnt = 0
        epoch = 1
        get_train_minibatch = examples.TrainingMinibatchStream()
        logging.info("INITIALIZING TRAINING STATE")

    while True:
        logging.info("STARTING EPOCH #%d" % epoch)
        for ebatch in get_train_minibatch:
            cnt += len(ebatch)
            m.train(ebatch)

            if cnt % (int(1000./HYPERPARAMETERS["MINIBATCH SIZE"])*HYPERPARAMETERS["MINIBATCH SIZE"]) == 0:
                logging.info("Finished training step %d (epoch %d)" % (cnt, epoch))
            if cnt % (int(100000./HYPERPARAMETERS["MINIBATCH SIZE"])*HYPERPARAMETERS["MINIBATCH SIZE"]) == 0:
                if os.path.exists(os.path.join(rundir, "BAD")):
                    logging.info("Detected file: %s\nSTOPPING" % os.path.join(rundir, "BAD"))
                    sys.stderr.write("Detected file: %s\nSTOPPING\n" % os.path.join(rundir, "BAD"))
                    sys.exit(0)
            if cnt % (int(HYPERPARAMETERS["VALIDATE_EVERY"]*1./HYPERPARAMETERS["MINIBATCH SIZE"])*HYPERPARAMETERS["MINIBATCH SIZE"]) == 0:
                state.save(m, cnt, epoch, get_train_minibatch, rundir, newkeystr)
        get_train_minibatch = examples.TrainingMinibatchStream()
        epoch += 1
