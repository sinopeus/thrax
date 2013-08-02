"""
Theano graph of Collobert & Weston language model.
Originally written by Joseph Turian, adapted for Python 3 by Xavier Goás Aguililla.
"""

import theano, logging
from theano.compile import pfunc
floatX = theano.configparser.parse_config_string('scalar.floatX')

from theano import tensor as t
#from theano import scalar as s

from theano.tensor.basic import horizontal_stack, dot
from theano.gradient import grad as gradient

import theano.compile

COMPILE_MODE = theano.compile.mode.Mode('c|py', 'fast_run')

import numpy

def activation_function(r):
    return t.tanh(r)

def stack(x):
    assert len(x) >= 2
    return horizontal_stack(*x)

class Graph:
    def __init__(self, hyperparameters, parameters):
        self.hidden_weights = parameters.hidden_weights
        self.hidden_biases = parameters.hidden_biases
        self.output_weights = parameters.output_weights
        self.output_biases = parameters.output_biases
        self.hyperparameters = hyperparameters
        self.cached_functions = {}

    def score(self,x):
        prehidden = dot(x, self.hidden_weights) + self.hidden_biases
        hidden = activation_function(prehidden)
        score = dot(hidden, self.output_weights) + self.output_biases
        return score, prehidden

    def predict(self, correct_sequence, parameters):
        fn = self.functions(sequence_length=len(correct_sequence))[0]
        r = fn(*(correct_sequence))
        assert len(r) == 1
        r = r[0]
        assert r.shape == (1, 1)
        return r[0,0]

    def train(self, correct_sequence, noise_sequence, learning_rate):
        assert len(correct_sequence) == len(noise_sequence)
        fn = self.functions(sequence_length=len(correct_sequence))[1]
        r = fn(*(correct_sequence + noise_sequence + [learning_rate]))
        dcorrect_inputs = r[:len(correct_sequence)]
        r = r[len(correct_sequence):]
        dnoise_inputs = r[:len(noise_sequence)]
        r = r[len(correct_sequence):]
        (loss, unpenalized_loss, l1penalty, correct_score, noise_score) = r

        return (dcorrect_inputs, dnoise_inputs, loss, unpenalized_loss, l1penalty, correct_score, noise_score)

    def verbose_predict(self, correct_sequence, parameters):
        fn = self.functions(sequence_length=len(correct_sequence))[2]
        r = fn(*(correct_sequence))
        assert len(r) == 2
        (score, prehidden) = r
        assert score.shape == (1, 1)
        return score[0,0], prehidden

    def functions(self, sequence_length):
        cachekey = (sequence_length)

        if cachekey not in self.cached_functions:
            logging.info("Need to construct graph for sequence_length=%d..." % (sequence_length))

            correct_inputs = [t.xmatrix() for i in range(sequence_length)]
            noise_inputs = [t.xmatrix() for i in range(sequence_length)]
            learning_rate = t.xscalar()

            stacked_correct_inputs = stack(correct_inputs)
            stacked_noise_inputs = stack(noise_inputs)

            correct_score, correct_prehidden = self.score(stacked_correct_inputs)
            noise_score, noise_prehidden = self.score(stacked_noise_inputs)
            unpenalized_loss = t.clip(1 - correct_score + noise_score, 0, 1e999)

            l1penalty = t.as_tensor_variable(numpy.asarray(0, dtype=floatX))
            loss = (unpenalized_loss.T + l1penalty).T

            total_loss = t.sum(loss)

            (dhidden_weights, dhidden_biases, doutput_weights, doutput_biases) = gradient(total_loss, [self.hidden_weights, self.hidden_biases, self.output_weights, self.output_biases])
            dcorrect_inputs = gradient(total_loss, correct_inputs)
            dnoise_inputs = gradient(total_loss, noise_inputs)
            predict_inputs = correct_inputs
            train_inputs = correct_inputs + noise_inputs + [learning_rate]
            verbose_predict_inputs = predict_inputs
            predict_outputs = [correct_score]
            train_outputs = dcorrect_inputs + dnoise_inputs + [loss, unpenalized_loss, l1penalty, correct_score, noise_score]
            verbose_predict_outputs = [correct_score, correct_prehidden]

            import theano.gof.graph

            nnodes = len(theano.gof.graph.ops(predict_inputs, predict_outputs))
            logging.info("About to compile prediction function over %d ops [nodes]..." % nnodes)
            predict_function = pfunc(predict_inputs, predict_outputs, mode=COMPILE_MODE)
            logging.info("...done constructing graph for sequence_length=%d" % (sequence_length))

            nnodes = len(theano.gof.graph.ops(verbose_predict_inputs, verbose_predict_outputs))
            logging.info("About to compile verbose prediction function over %d ops [nodes]..." % nnodes)
            verbose_predict_function = pfunc(verbose_predict_inputs, verbose_predict_outputs, mode=COMPILE_MODE)
            logging.info("...done constructing graph for sequence_length=%d" % (sequence_length))

            nnodes = len(theano.gof.graph.ops(train_inputs, train_outputs))
            logging.info("About to compile training function over %d ops [nodes]..." % nnodes)
            train_function = pfunc(train_inputs, train_outputs, mode=COMPILE_MODE, updates=[(p, p-learning_rate*gp) for p, gp in zip((self.hidden_weights, self.hidden_biases, self.output_weights, self.output_biases), (dhidden_weights, dhidden_biases, doutput_weights, doutput_biases))])
            logging.info("...done constructing graph for sequence_length=%d" % (sequence_length))

            self.cached_functions[cachekey] = (predict_function, train_function, verbose_predict_function)

        return self.cached_functions[cachekey]
