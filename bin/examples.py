"""
Methods for getting examples.
"""

from collections import Iterator
class Validator:
    def __init__(self, validation_corpus, model):
        self.corpus, self.dictionary, self.hyperparameters = corpus, dictionary, hyperparameters
        
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

class ExampleStream:
    def __init__(self, corpus, dictionary, hyperparameters):
        self.corpus, self.dictionary, self.hyperparameters = corpus, dictionary, hyperparameters
        self.count = 0

    def __iter__(self):
        self.count = 0

        for sentence in self.corpus:
            prevwords = []
            for word in sentence:
                if self.dictionary.contains(word):
                    prevwords.append(self.dictionary.lookup(word))
                    if len(prevwords) >= self.hyperparameters.window_size:
                        self.count += 1
                        yield prevwords[-self.hyperparameters.window_size:]
                else:
                    prevwords = []

    def __getstate__(self):
        return count

    def __setstate__(self, count):
        logging.info("Fast-forwarding example stream to text window %s." % count)
        while count != self.count:
            next(self)
        logging.info("Back at text window %s." % count)

class BatchStream(Iterator):
    def __init__(self, stream):
        self.stream = stream

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        while len(batch) < self.hyperparameters.batch_size:
            batch.append(next(self.stream))
        return batch
