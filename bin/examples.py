"""
Methods for getting examples.
"""

import lexicon

class TrainingExampleStream(object):
    def __init__(self, corpus, dictionary, hyperparameters):
        self.corpus, self.dictionary, self.hyperparameters = corpus, dictionary, hyperparameters
        self.count = 0
    
    def __iter__(self):
        self.count = 0

        for sentence in iter(self.training_corpus):
            prevwords = []
            for word in sentence:
                if self.dictionary.contains(word):
                    prevwords.append(self.dictionary.lookup(word))
                    if len(prevwords) >= self.hyperparameters.window_size:
                        self.count += 1
                        yield prevwords[-self.hyperparameters.window_size:]
                else:
                    prevwords = []

    def get_batch(self):
        batch = []

        while len(batch) < self.hyperparameters.batch_size:
            batch.append(next(self))

        return batch

    def get_validation_example(self):
        for sentence in iter(open(self.hyperparameters.validation_sentences)):
            prevwords = []
            for word in sentence:
                if self.dictionary.contains(word):
                    prevwords.append(self.dictionary.lookup(word))
                    if len(prevwords) >= self.hyperparameters.window_size:
                        yield prevwords[-self.hyperparameters.window_size:]
                    else:
                        prevwords = []

    def __getstate__(self):
        return self.training_corpus, self.dictionary, self.window_size, self.batch_size, count

    def __setstate__(self, state):
        self.training_corpus, self.dictionary, self.window_size, self.batch_size, count = state
        logging.info("__setstate__(%s)..." % repr(state))
        logging.info(stats())
        while count != self.count:
            next(iter(self))
        logging.info("...__setstate__(%s)" % repr(state))
        logging.info(stats())
