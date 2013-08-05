import logging, os.path, pickle, validate

class TrainingState:
    def __init__(self, corpus, dictionary, hyperparameters):
        self.corpus, self.dictionary, self.hyperparameters = corpus, dictionary, hyperparameters

        logging.info("Initializing training state.")

        from model.model import Model
        self.model = Model(self.hyperparameters)

        self.count, self.epoch = (0, 1)

        from lexicon import Corpus
        from validate import Validator

        logging.info("Processing validation corpus ...")
        validation_corpus = Corpus(os.path.join(hyperparameters.data_dir, hyperparameters.validation_sentences))
        logging.info("Validation corpus processed.")

        logging.info("Initialising model validator...")
        self.validator = Validator(validation_corpus, self.model)
        logging.info("Model validator initialised.")

        from examples import ExampleStream, BatchStream

        logging.info("Initialising text window stream...")
        self.examples = ExampleStream(self.corpus, self.dictionary, self.hyperparameters)
        logging.info("Text window stream initialised.")

        logging.info("Initialising batch stream...")
        self.batches = BatchStream(self.examples)
        logging.info("Batch stream initialised.")

    def run_epoch(self):
        logging.info("Starting epoch #%d." % self.epoch)

        for batch in self.batches: self.process(batch)

        self.epoch += 1

        logging.info("Finished epoch #%d. Rewinding training stream." % self.epoch)
        self.corpus.rewind()

        from examples import ExampleStream
        self.stream = ExampleStream(self.corpus, self.dictionary, self.hyperparameters)

    def process(self, batch):
        self.count += len(batch)
        self.model.train(batch)

        if self.count % (int(1000. / self.hyperparameters.batch_size) * self.hyperparameters.batch_size) == 0:
            logging.info("Finished training step %d (epoch %d)" % (self.count, self.epoch))

        if self.count % (int( self.hyperparameters.validate_every * 1./self.hyperparameters.batch_size ) * self.hyperparameters.batch_size) == 0:
            self.save()
            self.validator.validate(self.count)

    def save(self):
        filename = os.path.join(self.hyperparameters.run_dir, "trainstate.pkl")
        logging.info("Trying to save training state to %s..." % filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.__getstate__(), f)

    def __getstate__(self):
        return (self.corpus.__getstate__(), self.dictionary.__getstate__(), self.hyperparameters, self.model.__getstate__(), self.count, self.epoch, self.stream.__getstate__())

    def __setstate__(self, state):
        from model.model import Model
        self.model = Model(self.hyperparameters)
        self.model.__setstate__(state[-4])
        self.count, self.epoch = state[-3:-1]
        self.stream.__setstate__(state[-1])
