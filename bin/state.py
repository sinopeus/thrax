"""
A class for keeping the state of training. A separate function is provided for loading a state.
"""

import logging, os.path, pickle, sys, examples
from stats import stats

def load(rundir=None):
    filename = os.path.join(rundir, "trainstate.pkl")
    trainstate = pickle.load(open(filename))
    trainstate.rundir = rundir
    return trainstate

class TrainingState:
    def __init__(self, rundir=None, training_sentences, word_hash, window_size):
        import model.model as md
        self.model = md.Model()
        self.count = 0
        self.epoch = 1
        self.stream = examples.TrainingStream(training_sentences,word_hash,window_size)
        self.batch_size = batch_size
        self.batch = examples.TrainingMinibatchStream(self.stream, self.batch_size)
        self.validate_every = validate_every
        self.rundir = rundir

    def epoch(self):
        logging.info("STARTING EPOCH #%d" % self.epoch)
        for ebatch in self.batch:
            self.count += len(ebatch)
            self.model.train(ebatch)

            if self.count % (int(1000. / batch_size) * batch_size) == 0:
                logging.info("Finished training step %d (epoch %d)" % (self.count, self.epoch))
                
            if self.count % (int(100000. / batch_size) * batch_size) == 0:
                if os.path.exists(os.path.join(self.rundir, "BAD")):
                    logging.info("Detected file: %s\nSTOPPING" % os.path.join(self.rundir, "BAD"))
                    sys.stderr.write("Detected file: %s\nSTOPPING\n" % os.path.join(self.rundir, "BAD"))
                    sys.exit(0)
                    
            if self.count % (int( validate_every * 1./batch_size ) * batch_size) == 0:
                self.save()
                
        self.batch = examples.TrainingMinibatchStream()
        self.epoch += 1
        
    def save(self):
        filename = os.path.join(self.rundir, "model-%d.pkl" % self.count)

        try:
            logging.info("Removing old model %s..." % filename)
            os.remove(filename)
            logging.info("...removed %s" % filename)
        except IOError:
            logging.info("Could NOT remove %s" % filename)
            filename = os.path.join(self.rundir, "model.pkl")

        logging.info("Writing model to %s..." % filename)
        logging.info(stats())
        pickle.dump(self.model, open(filename, "wb"))
        logging.info("...done writing model to %s" % filename)
        logging.info(stats())

        filename = os.path.join(self.rundir, "trainstate-%d.pkl" % self.count)

        try:
            logging.info("Removing old training state %s..." % filename)
            os.remove(filename)
            logging.info("...removed %s" % filename)
        except IOError:
            logging.info("Could NOT remove %s" % filename)
            filename = os.path.join(self.rundir, "trainstate.pkl")

        logging.info("Writing training state to %s..." % filename)
        logging.info(stats())
        pickle.dump(self, open(filename, "wb"))
        logging.info("...done writing training state to %s" % filename)
        logging.info(stats())
