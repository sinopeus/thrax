"""
Methods for getting examples.
"""
import string, logging
from stats import stats

class TrainingExampleStream(object):
    def __init__(self, **kwargs):
        for (k, v) in kwargs.items():
            setattr(self, k, v)
        self.count = 0
    
    def __iter__(self):
        self.count = 0

        for sentence in iter(self.training_corpus):
            prevwords = []
            for word in sentence:
                if self.dictionary.contains(word):
                    prevwords.append(self.dictionary.lookup(word))
                    if len(prevwords) >= self.window_size:
                        self.count += 1
                        yield prevwords[-self.window_size:]
                else:
                    prevwords = []

    def get_batch():
        batch = []
        for e in iter(self):
            batch.append(e)
            if len(batch) >= self.batch_size:
                assert len(batch) == self.batch_size
                yield batch
                batch = []

    def get_validation_example(self):
        for sentence in iter(open(self.validation_sentences)):
            prevwords = []
            for word in sentence:
                if dictionary.contains(word):
                    prevwords.append(self.dictionary.lookup(w))
                    if len(prevwords) >= window_size:
                        yield prevwords[-window_size:]
                    else:
                        prevwords = []

    # def __getstate__(self):
    #     return self

    # def __setstate__(self, state):
    #     self.training_corpus, self.dictionary, self.window_size, self.batch_size, count = state
    #     logging.info("__setstate__(%s)..." % repr(state))
    #     logging.info(stats())
    #     iter = self.__iter__()
    #     while count != self.count:
    #         iter.next()
    #     logging.info("...__setstate__(%s)" % repr(state))
    #     logging.info(stats())


# class TrainingMinibatchStream(object):
#     def __init__(self, stream, batch_size):
#         self.stream = stream
#         self.batch_size = batch_size
    
#     def __iter__(self):
#         minibatch = []
#         for e in self.stream:
#             minibatch.append(e)
#             if len(minibatch) >= self.batch_size:
#                 assert len(minibatch) == self.batch_size
#                 yield minibatch
#                 minibatch = []

#     def __getstate__(self):
#         return (self.stream, self.batch_size)

#     def __setstate__(self, state):
#         (self.stream, self.batch_size) = state

