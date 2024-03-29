"""
Theano graph of Collobert & Weston language model.
Originally written by Joseph Turian, adapted for Python 3 by Xavier Goás Aguililla.
"""

import theano
from theano.compile import pfunc, sharedvalue
floatX = theano.configparser.parse_config_string('scalar.floatX')

from theano import tensor as t
from theano import scalar as s

from theano.tensor.basic import horizontal_stack, dot
from theano import gradient

import theano.compile

COMPILE_MODE = theano.compile.mode.Mode('c|py', 'fast_run')

import numpy

def activation_function(r):
    return t.tanh(r)

def stack(x):
    assert len(x) >= 2
    return horizontal_stack(*x)

def score(x):
    prehidden = dot(x, hidden_weights) + hidden_biases
    hidden = activation_function(prehidden)
    score = dot(hidden, output_weights) + output_biases
    return score, prehidden

cached_functions = {}

def functions(sequence_length):
    """
    Return two functions
     * The first function does prediction.
     * The second function does learning.
    """
    global cached_functions
    cachekey = (sequence_length)
    if len(cached_functions.keys()) > 1:
        # This is problematic because we use global variables for the model parameters.
        # Hence, we might be unsafe, if we are using the wrong model parameters globally.
        assert 0
    if cachekey not in cached_functions:
        logging.info("Need to construct graph for sequence_length=%d..." % (sequence_length))
        learning_rate = t.xscalar()
        inputs = [t.matrix() for i in range(sequence_length)]
        correct_outputs = [t.vector() for i in range(sequence_length)]
        stacked_inputs = stack(inputs)
        output, prehidden = compute(stacked_inputs)

        loss =  

        total_loss = t.sum(loss)

        (dhidden_weights, dhidden_biases, doutput_weights, doutput_biases) = t.grad(total_loss, [hidden_weights, hidden_biases, output_weights, output_biases])
        dcorrect_inputs = t.grad(total_loss, correct_inputs)
        dnoise_inputs = t.grad(total_loss, noise_inputs)
        predict_inputs = correct_inputs
        train_inputs = correct_inputs + noise_inputs + [learning_rate]
        verbose_predict_inputs = predict_inputs
        predict_outputs = [correct_score]
        train_outputs = dcorrect_inputs + dnoise_inputs + [loss, unpenalized_loss, l1penalty, correct_score, noise_score]
        verbose_predict_outputs = [correct_score, correct_prehidden]

        import theano.gof.graph

        nnodes = len(theano.gof.graph.ops(predict_inputs, predict_outputs))
        logging.info("About to compile predict function over %d ops [nodes]..." % nnodes)
        predict_function = pfunc(predict_inputs, predict_outputs, mode=COMPILE_MODE)
        logging.info("...done constructing graph for sequence_length=%d" % (sequence_length))

        nnodes = len(theano.gof.graph.ops(verbose_predict_inputs, verbose_predict_outputs))
        logging.info("About to compile predict function over %d ops [nodes]..." % nnodes)
        verbose_predict_function = pfunc(verbose_predict_inputs, verbose_predict_outputs, mode=COMPILE_MODE)
        logging.info("...done constructing graph for sequence_length=%d" % (sequence_length))

        nnodes = len(theano.gof.graph.ops(train_inputs, train_outputs))
        logging.info("About to compile train function over %d ops [nodes]..." % nnodes)
        train_function = pfunc(train_inputs, train_outputs, mode=COMPILE_MODE, updates=[(p, p-learning_rate*gp) for p, gp in zip((hidden_weights, hidden_biases, output_weights, output_biases), (dhidden_weights, dhidden_biases, doutput_weights, doutput_biases))])
        logging.info("...done constructing graph for sequence_length=%d" % (sequence_length))

        cached_functions[cachekey] = (predict_function, train_function, verbose_predict_function)
    return cached_functions[cachekey]

def predict(correct_sequence, parameters):
    fn = functions(sequence_length=len(correct_sequence))[0]
    r = fn(*(correct_sequence))
    assert len(r) == 1
    r = r[0]
    assert r.shape == (1, 1)
    return r[0,0]

def verbose_predict(correct_sequence, parameters):
    fn = functions(sequence_length=len(correct_sequence))[2]
    r = fn(*(correct_sequence))
    assert len(r) == 2
    (score, prehidden) = r
    assert score.shape == (1, 1)
    return score[0,0], prehidden

def train(correct_sequence, noise_sequence, learning_rate):
    assert len(correct_sequence) == len(noise_sequence)
    fn = functions(sequence_length=len(correct_sequence))[1]
    r = fn(*(correct_sequence + noise_sequence + [learning_rate]))
    dcorrect_inputs = r[:len(correct_sequence)]
    r = r[len(correct_sequence):]
    dnoise_inputs = r[:len(noise_sequence)]
    r = r[len(correct_sequence):]
    (loss, unpenalized_loss, l1penalty, correct_score, noise_score) = r

    return (dcorrect_inputs, dnoise_inputs, loss, unpenalized_loss, l1penalty, correct_score, noise_score)
