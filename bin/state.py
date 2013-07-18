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
    def __init__(self, **kwargs):
        for (k, v) in kwargs.items():
            setattr(self, k, v)
        
        import model.model as md
        self.model = md.Model(self.window_size, self.dictionary.size(), self.embedding_size, self.hidden_size, 0)
        self.count = 0
        self.epoch = 1
        self.stream = examples.TrainingStream(training_corpus = self.corpus, dictionary = self.dictionary, window_size = self.window_size, batch_size = self.batch_size, validation_sentences = self.validation_sentences)
        self.batch = self.stream.get_batch()

    def epoch(self):
        logging.info("STARTING EPOCH #%d" % self.epoch)
        for example in self.batch:
            self.count += len(example)
            self.model.train(example)

            if self.count % (int(1000. / self.batch_size) * self.batch_size) == 0:
                logging.info("Finished training step %d (epoch %d)" % (self.count, self.epoch))
                
            if self.count % (int(100000. / self.batch_size) * self.batch_size) == 0:
                if os.path.exists(os.path.join(self.rundir, "BAD")):
                    logging.info("Detected file: %s\nSTOPPING" % os.path.join(self.rundir, "BAD"))
                    sys.stderr.write("Detected file: %s\nSTOPPING\n" % os.path.join(self.rundir, "BAD"))
                    sys.exit(0)
                    
            if self.count % (int( self.validate_every * 1./self.batch_size ) * self.batch_size) == 0:
                self.save()
                
        self.batch = self.stream.get_batch()
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
