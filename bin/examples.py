"""
Methods for getting examples.
"""

import stats, string, sys
from hyperparameters import *

class TrainingExampleStream(object):
    def __init__(self):
        self.count = 0
        pass
    
    def __iter__(self):
        from vocabulary import word_hash
        self.filename = TRAIN_SENTENCES
        self.count = 0

        for l in iter(open(self.filename)):
            prevwords = []
            for w in string.split(l):
                w = string.strip(w)
                id = None
                if word_hash.exists(w):
                    prevwords.append(word_hash.id(w))
                    if len(prevwords) >= WINDOW_SIZE:
                        self.count += 1
                        yield prevwords[-WINDOW_SIZE:]
                else:
                    prevwords = []

    def __getstate__(self):
        return self.filename, self.count

    def __setstate__(self, state):
        """
        @warning: We ignore the filename.  If we wanted
        to be really fastidious, we would assume that
        HYPERPARAMETERS["TRAIN_SENTENCES"] might change.  The only
        problem is that if we change filesystems, the filename
        might change just because the base file is in a different
        path. So we issue a warning if the filename is different from
        """
        filename, count = state
        logging.info("__setstate__(%s)..." % repr(state))
        logging.info(stats())
        iter = self.__iter__()
        while count != self.count:
            iter.next()
        if self.filename != filename:
            assert self.filename == TRAIN_SENTENCES
            logging.info("self.filename %s != filename given to __setstate__ %s" % (self.filename, filename))
        logging.info("...__setstate__(%s)" % repr(state))
        logging.info(stats())

class TrainingMinibatchStream(object):
    def __init__(self):
        pass
    
    def __iter__(self):
        minibatch = []
        self.get_train_example = TrainingExampleStream()
        for e in self.get_train_example:
            minibatch.append(e)
            if len(minibatch) >= MINIBATCH_SIZE:
                assert len(minibatch) == MINIBATCH_SIZE
                yield minibatch
                minibatch = []

    def __getstate__(self):
        return (self.get_train_example.__getstate__(),)

    def __setstate__(self, state):
        """
        @warning: We ignore the filename.
        """
        self.get_train_example = TrainingExampleStream()
        self.get_train_example.__setstate__(state[0])

def get_validation_example():
    from vocabulary import word_hash
    for l in iter(open(VALIDATION_SENTENCES)):
        prevwords = []
        for w in string.split(l):
            w = string.strip(w)
            if word_hash.exists(w):
                prevwords.append(word_hash.id(w))
                if len(prevwords) >= WINDOW_SIZE:
                    yield prevwords[-WINDOW_SIZE:]
            else:
                prevwords = []
