import numpy, logging

class Validator:
    def __init__(self, validation_corpus, model):
        self.corpus = validation_corpus
        self.model = model

    def __iter__(self):
        for sentence in self.corpus:
            prevwords = []
            for word in sentence:
                if self.dictionary.contains(word):
                    prevwords.append(self.dictionary.lookup(word))
                    if len(prevwords) >= self.hyperparameters.window_size:
                        yield prevwords[-self.hyperparameters.window_size:]
                else:
                    prevwords = []

    def validate(self, cnt):
        import math
        logranks = []
        append = logranks.append # from here on we create shortcuts to functions, we want to speed up the validation as much as possible since it wastes valuable training time
        validate = self.model.validate
        log = math.log
        logging.info("Beginning validation at training step %d." % cnt)
        [append(lgr) for lgr in [log(score) for score in [validate(window) for window in self]]] # list comprehension magic
        num_logranks = numpy.array(logranks)
        logging.info("Validation at training step %d: mean(logrank) = %.2f, stddev(logrank) = %.2f, cnt = %d", (cnt, numpy.mean(num_logranks), numpy.std(num_logranks), len(logranks)))
