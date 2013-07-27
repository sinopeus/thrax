"""
A class for keeping the state of training. A separate function is provided for loading a state.
"""

import logging, os.path, pickle, sys
from stats import stats

def load(rundir=None):
    logging.info("Trying to read training state from %s..." % rundir)
    filename = os.path.join(rundir, "trainstate.pkl")
    with open(filename, 'rb') as f:
        saved_state = pickle.load(f)
    corpus, dictionary, hyperparameters = saved_state[0:2]

    trainstate = TrainingState(corpus, dictionary, hyperparameters)
    trainstate.__setstate__(saved_state)
    trainstate.rundir = rundir
    return trainstate

class TrainingState:
    def __init__(self, corpus, dictionary, hyperparameters):
        self.corpus, self.dictionary, self.hyperparameters = corpus, dictionary, hyperparameters

        logging.info("INITIALIZING TRAINING STATE")

        from model.model import Model
        self.model = Model(self.hyperparameters)
        self.count = 0
        self.epoch = 1

        from examples import TrainingExampleStream
        self.stream = TrainingExampleStream(self.corpus, self.dictionary, self.hyperparameters)

    def epoch(self):
        logging.info("Starting epoch #%d." % self.epoch)

        batch = self.stream.get_batch()

        while batch:
            self.process(batch)
            batch = self.stream.get_batch()

        self.epoch += 1
        logging.info("Finished epoch #%d. Rewinding training stream." % self.epoch)
        self.corpus.rewind()
        self.stream = TrainingExampleStream(self.corpus, self.dictionary, self.hyperparameters)

    def process(self, batch):
        for example in batch:
            self.count += len(example)
            self.model.train(example)

            if self.count % (int(1000. / self.hyperparameters.batch_size) * self.hyperparameters.batch_size) == 0:
                logging.info("Finished training step %d (epoch %d)" % (self.count, self.epoch))

            if self.count % (int(100000. / self.hyperparameters.batch_size) * self.hyperparametersbatch_size) == 0:
                if os.path.exists(os.path.join(self.hyperparameters.run_dir, "BAD")):
                    logging.info("Detected file: %s\nSTOPPING" % os.path.join(self.hyperparameters.run_dir, "BAD"))
                    sys.stderr.write("Detected file: %s\nSTOPPING\n" % os.path.join(self.hyperparameters.run_dir, "BAD"))
                    sys.exit(0)

            if self.count % (int( self.hyperparameters.validate_every * 1./self.hyperparameters.batch_size ) * self.hyperparameters.batch_size) == 0:
                self.save()

    def save(self):
        filename = os.path.join(self.hyperparameters.run_dir, "trainstate.pkl")
        logging.info("Trying to save training state to %s..." % filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.__getstate__(), f)

    def __getstate__(self):
        return (self.corpus, self.dictionary, self.hyperparameters, self.model.__getstate__(), self.count, self.epoch, self.stream.__getstate__())

    def __setstate__(self, state):
        self.model = Model(self.hyperparameters)
        self.model.__setstate__(state[-4])
        self.count, self.epoch = state[-3:-2]
        self.stream.__setstate__(state[-1])
