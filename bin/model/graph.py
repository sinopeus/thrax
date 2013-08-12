"""
Theano graph of Collobert & Weston language model.
Originally written by Joseph Turian, adapted for Python 3 by Xavier Goás Aguililla.
"""

import theano, logging, numpy
import theano.tensor.basic as t
from theano.gradient import grad

theano.config.floatX = 'float32'
floatX = theano.config.floatX
COMPILE_MODE = "FAST_RUN"

class Graph:
    def __init__(self, hyperparameters, parameters):
        self.hyperparameters = hyperparameters
        self.parameters = parameters
        self.cache = {}

    def score(self,x):
        prehidden = t.dot(x, self.parameters.hidden_weights) + self.parameters.hidden_biases
        hidden = t.clip(prehidden, -1, 1)
        score = t.dot(hidden, self.parameters.output_weights) + self.parameters.output_biases
        return score, prehidden

    def predict(self, correct_sequence, parameters):
        f = self.functions(sequence_length=len(correct_sequence))[0]
        return f(correct_sequence)

    def train(self, correct_sequence, noise_sequence, learning_rate):
        assert len(correct_sequence) == len(noise_sequence)
        f = self.functions(sequence_length=len(correct_sequence))[1]
        return f(correct_sequence, noise_sequence, learning_rate)

    def verbose_predict(self, correct_sequence, parameters):
        f = self.functions(sequence_length=len(correct_sequence))[2]
        return f(correct_sequence)

    def functions(self, sequence_length):
        key = (sequence_length)

        if key not in self.cache:
            logging.info("Need to construct graph for sequence_length=%d..." % (sequence_length))

            # creating network input variable nodes
            correct_inputs = t.ftensor3("correct input")
            noise_inputs = t.ftensor3("noise input")
            learning_rate = t.fscalar("learning rate")

            # creating op nodes for firing the network
            correct_score, correct_prehidden = self.score(correct_inputs)
            noise_score, noise_prehidden = self.score(noise_inputs)

            # creating op nodes for the pairwise ranking cost function
            loss = t.clip(1 - correct_score + noise_score, 0, 1e999)
            total_loss = t.sum(loss)

            # the necessary cost function gradients
            parameters_gradient = grad(total_loss, list(self.parameters))
            correct_inputs_gradient = grad(total_loss, correct_inputs)
            noise_inputs_gradient = grad(total_loss, noise_inputs)

            # setting network inputs
            predict_inputs = [correct_inputs]
            train_inputs = [correct_inputs, noise_inputs, learning_rate]
            verbose_predict_inputs = predict_inputs

            # setting network outputs
            predict_outputs = [correct_score]
            train_outputs = [correct_inputs_gradient, noise_inputs_gradient, loss, correct_score, noise_score]
            verbose_predict_outputs = [correct_score, correct_prehidden]

            nnodes = len(theano.gof.graph.ops(predict_inputs, predict_outputs))
            logging.info("About to compile prediction function over %d ops [nodes]..." % nnodes)
            predict = theano.function(predict_inputs, predict_outputs, mode=COMPILE_MODE)
            logging.info("...done constructing graph for sequence_length=%d" % (sequence_length))

            nnodes = len(theano.gof.graph.ops(verbose_predict_inputs, verbose_predict_outputs))
            logging.info("About to compile verbose prediction function over %d ops [nodes]..." % nnodes)
            verbose_predict = theano.function(verbose_predict_inputs, verbose_predict_outputs, mode=COMPILE_MODE)
            logging.info("...done constructing graph for sequence_length=%d" % (sequence_length))

            nnodes = len(theano.gof.graph.ops(train_inputs, train_outputs))
            logging.info("About to compile training function over %d ops [nodes]..." % nnodes)
            train = theano.function(train_inputs, train_outputs, mode=COMPILE_MODE, updates=[(p, p - learning_rate * gp) for p, gp in zip(list(self.parameters), parameters_gradient)])
            logging.info("...done constructing graph for sequence_length=%d" % (sequence_length))

            self.cache[key] = (predict, train, verbose_predict)

        return self.cache[key]
