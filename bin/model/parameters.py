import theano.configparser
from theano.compile import sharedvalue
from math import pow, sqrt

floatX = theano.configparser.parse_config_string('scalar.floatX')
sqrt3 = sqrt(3.0)

class Parameters:
    import lexicon

    def __init__(self, window_size, vocab_size, vect_size, hidden_size, rnd_seed):
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.vect_size = vect_size
        self.hidden_size = hidden_size
        self.output_size = 1
        self.input_size = vect_size * vocab_size

        import numpy
        numpy.random.seed(rnd_seed)

        self.embeddings = numpy.asarray((numpy.random.rand(self.vocab_size, embedding_size) - 0.5)* 2 * 0.01, dtype=floatX)
        self.hidden_weights = shared(numpy.asarray(random_weights(self.input_size, self.hidden_size, scale_by=1), dtype=floatx))
        self.output_weights = shared(numpy.asarray(random_weights(self.hidden_size, self.output_size, scale_by=1), dtype=floatx))
        self.hidden_biases = shared(numpy.asarray(numpy.zeros((self.hidden_size,)), dtype=floatx))
        self.output_biases = shared(numpy.asarray(numpy.zeros((self.output_size,)), dtype=floatx))

    def normalize(self, indices):
        import numpy
        l2norm = numpy.square(self.embeddings[indices]).sum(axis=1)
        l2norm = numpy.sqrt(l2norm.reshape((len(indices),1)))
        self.embeddings[indices] /= l2norm
        import math
        self.embeddings[indices] *= math.sqrt(self.embeddings.shape[1])
    
def random_weights(nin, nout, scale_by=1./sqrt3, power=0.5):
    return (numpy.random.rand(nin, nout) * 2.0 - 1) * scale_by * sqrt3 / pow(nin,power) 
