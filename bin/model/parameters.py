import numpy, math, theano.configparser
from theano.compile.sharedvalue import shared

theano.config.floatX = 'float32'
floatX = theano.config.floatX
sqrt3 = math.sqrt(3.0)

class Parameters:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        numpy.random.seed()

        self.embeddings = numpy.asarray((numpy.random.rand(self.hyperparameters.vocab_size, self.hyperparameters.embedding_size) - 0.5)* 2 * 0.01, dtype=floatX)
        self.hidden_weights = shared(numpy.asarray(random_weights(self.hyperparameters.input_size, self.hyperparameters.hidden_size, scale_by=1), dtype=floatX))
        self.output_weights = shared(numpy.asarray(random_weights(self.hyperparameters.hidden_size, self.hyperparameters.output_size, scale_by=1), dtype=floatX))
        self.hidden_biases = shared(numpy.asarray(numpy.zeros((self.hyperparameters.hidden_size,)), dtype=floatX))
        self.output_biases = shared(numpy.asarray(numpy.zeros((self.hyperparameters.output_size,)), dtype=floatX))

    def __iter__(self):
        for param in (self.hidden_weights, self.output_weights, self.hidden_biases, self.output_biases): yield param

    def normalize(self, indices):
        l2norm = numpy.square(self.embeddings[indices]).sum(axis=1)
        l2norm = numpy.sqrt(l2norm.reshape((len(indices),1)))
        self.embeddings[indices] /= l2norm
        import math
        self.embeddings[indices] *= math.sqrt(self.embeddings.shape[1])

def random_weights(nin, nout, scale_by=1./sqrt3, power=0.5):
    return (numpy.random.rand(nin, nout) * 2.0 - 1) * scale_by * sqrt3 / math.pow(nin,power)
