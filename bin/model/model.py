from model.parameters import Parameters
import graph

import math
import logging

class Model:
    """
    A Model can:

    @type parameters: L{Parameters}
    @todo: Document
    """

    def __init__(self):
        self.parameters = Parameters()
        self.trainer = Trainer()
        graph.hidden_weights = self.parameters.hidden_weights
        graph.hidden_biases = self.parameters.hidden_biases
        graph.output_weights = self.parameters.output_weights
        graph.output_biases = self.parameters.output_biases

    def __getstate__(self):
        return (self.parameters, self.trainer)

    def __setstate__(self, state):
        (self.parameters, self.trainer) = state
        graph.hidden_weights = self.parameters.hidden_weights
        graph.hidden_biases = self.parameters.hidden_biases
        graph.output_weights = self.parameters.output_weights
        graph.output_biases = self.parameters.output_biases

    def embed(self, sequence):
        """
        Embed a sequence of vocabulary IDs
        """
        seq = [self.parameters.embeddings[s] for s in sequence]
        import numpy
        return [numpy.resize(s, (1, s.size)) for s in seq]

    def embeds(self, sequences):
        """
        Embed sequences of vocabulary IDs.
        If we are given a list of MINIBATCH lists of SEQLEN items, return a list of SEQLEN matrices of shape (MINIBATCH, EMBSIZE)
        """
        embs = []
        for sequence in sequences:
            embs.append(self.embed(sequence))

        for emb in embs: assert len(emb) == len(embs[0])

        new_embs = []
        for i in range(len(embs[0])):
            colembs = [embs[j][i] for j in range(len(embs))]
            import numpy
            new_embs.append(numpy.vstack(colembs))
            assert new_embs[-1].shape == (len(sequences), self.parameters.embedding_size)
        assert len(new_embs) == len(sequences[0])
        return new_embs

    def corrupt_example(self, e):
        """
        Return a corrupted version of example e, plus the weight of this example.
        """
        import random
        import copy
        e = copy.copy(e)
        last = e[-1]
        cnt = 0
        while e[-1] == last:
            e[-1] = random.randint(0, self.parameters.vocab_size-1)
            pr = 1./self.parameters.vocab_size
            cnt += 1
            # Backoff to 0gram smoothing if we fail 10 times to get noise.
            if cnt > 10: e[-1] = random.randint(0, self.parameters.vocab_size-1)
        weight = 1./pr
        return e, weight

    def corrupt_examples(self, correct_sequences):
        noise_sequences = []
        weights = []
        for e in correct_sequences:
            noise_sequence, weight = self.corrupt_example(e)
            noise_sequences.append(noise_sequence)
            weights.append(weight)
        return noise_sequences, weights

    def train(self, correct_sequences):
        learning_rate = LEARNING_RATE
        noise_sequences, weights = self.corrupt_examples(correct_sequences)
        # All weights must be the same, if we first multiply by the learning rate
        for w in weights: assert w == weights[0]

        r = graph.train(self.embeds(correct_sequences), self.embeds(noise_sequences), learning_rate * weights[0])
        (dcorrect_inputss, dnoise_inputss, losss, unpenalized_losss, l1penaltys, correct_scores, noise_scores) = r

        to_normalize = set()
        for ecnt in range(len(correct_sequences)):
            (loss, unpenalized_loss, correct_score, noise_score) = \
                (losss[ecnt], unpenalized_losss[ecnt], correct_scores[ecnt], noise_scores[ecnt])
            if l1penaltys.shape == ():
                assert l1penaltys == 0
                l1penalty = 0
            else:
                l1penalty = l1penaltys[ecnt]
            correct_sequence = correct_sequences[ecnt]
            noise_sequence = noise_sequences[ecnt]

            dcorrect_inputs = [d[ecnt] for d in dcorrect_inputss]
            dnoise_inputs = [d[ecnt] for d in dnoise_inputss]

            self.trainer.update(loss, correct_score, noise_score, unpenalized_loss, l1penalty)
                
            for w in weights: assert w == weights[0]
            embedding_learning_rate = EMBEDDING_LEARNING_RATE * weights[0]
            if loss == 0:
                for di in dcorrect_inputs + dnoise_inputs:
                    assert (di == 0).all()
    
            if loss != 0:
                for (i, di) in zip(correct_sequence, dcorrect_inputs):
                    assert di.shape == (self.parameters.embedding_size,)
                    self.parameters.embeddings[i] -= 1.0 * embedding_learning_rate * di
                    if NORMALIZE_EMBEDDINGS:
                        to_normalize.add(i)
                for (i, di) in zip(noise_sequence, dnoise_inputs):
                    assert di.shape == (self.parameters.embedding_size,)
                    self.parameters.embeddings[i] -= 1.0 * embedding_learning_rate * di
                    if NORMALIZE_EMBEDDINGS:
                        to_normalize.add(i)
                                
        if len(to_normalize) > 0:
            to_normalize = [i for i in to_normalize]
            self.parameters.normalize(to_normalize)

    def predict(self, sequence):
        (score) = graph.predict(self.embed(sequence), self.parameters)
        return score

    def verbose_predict(self, sequence):
        (score, prehidden) = graph.verbose_predict(self.embed(sequence), self.parameters)
        return score, prehidden

    def validate(self, sequence):
        """
        Get the rank of this final word, as opposed to all other words in the vocabulary.
        """
        import random
        r = random.Random()
        r.seed(0)

        import copy
        corrupt_sequence = copy.copy(sequence)
        rank = 1
        correct_score = self.predict(sequence)
        for i in range(self.parameters.vocab_size):
            if r.random() > VALIDATION_LOGRANK_NOISE_EXAMPLES_PERCENT: continue
            if i == sequence[-1]: continue
            corrupt_sequence[-1] = i
            corrupt_score = self.predict(corrupt_sequence)
            if correct_score <= corrupt_score:
                rank += 1
        return rank

class Trainer:
    def __init__(self):
        self.loss = MovingAverage() 
        self.err = MovingAverage()
        self.lossnonzero = MovingAverage()
        self.squashloss = MovingAverage()
        self.unpenalized_loss = MovingAverage()
        self.l1penalty  = MovingAverage()
        self.unpenalized_lossnonzero = MovingAverage()
        self.correct_score = MovingAverage()
        self.noise_score = MovingAverage()
        self.cnt = 0

    def update(self, loss, correct_score, noise_score, unpenalized_loss, l1penalty):
        self.loss.add(loss)
        self.err.add(int(correct_score <= noise_score))
        self.lossnonzero.add(int(loss > 0))
        squashloss = 1. / (1. + math.exp(-loss))
        self.squashloss.add(squashloss)
        self.unpenalized_loss.add(unpenalized_loss)
        self.l1penalty.add(l1penalty)
        self.unpenalized_lossnonzero.add(int(unpenalized_loss > 0))
        self.correct_score.add(correct_score)
        self.noise_score.add(noise_score)
        self.cnt += 1

        if self.cnt % 10000 == 0: update_log()

    def update_log(self):
        logging.info(("After %d updates, pre-update train loss %s" % (self.cnt, self.loss.verbose_string())))
        logging.info(("After %d updates, pre-update train error %s" % (self.cnt, self.err.verbose_string())))
        logging.info(("After %d updates, pre-update train Pr(loss != 0) %s" % (self.cnt, self.lossnonzero.verbose_string())))
        logging.info(("After %d updates, pre-update train squash(loss) %s" % (self.cnt, self.squashloss.verbose_string())))
        logging.info(("After %d updates, pre-update train unpenalized loss %s" % (self.cnt, self.unpenalized_loss.verbose_string())))
        logging.info(("After %d updates, pre-update train l1penalty %s" % (self.cnt, self.l1penalty.verbose_string())))
        logging.info(("After %d updates, pre-update train Pr(unpenalized loss != 0) %s" % (self.cnt, self.unpenalized_lossnonzero.verbose_string())))
        logging.info(("After %d updates, pre-update train correct score %s" % (self.cnt, self.correct_score.verbose_string())))
        logging.info(("After %d updates, pre-update train noise score %s" % (self.cnt, self.noise_score.verbose_string())))
        

class MovingAverage:
    def __init__(self, percent=False):
        self.mean = 0.
        self.variance = 0
        self.cnt = 0
        self.percent = percent

    def add(self, v):
        """
        Add value v to the moving average.
        """
        self.cnt += 1
        self.mean = self.mean - (2. / self.cnt) * (self.mean - v)
        # I believe I should compute self.variance AFTER updating the moving average, because
        # the estimate of the mean is better.
        # Yoshua concurs.
        this_variance = (v - self.mean) * (v - self.mean)
        self.variance = self.variance - (2. / self.cnt) * (self.variance - this_variance)

    def __str__(self):
        if self.percent:
            return "(moving average): mean=%.3f%% stddev=%.3f" % (self.mean, math.sqrt(self.variance))
        else:
            return "(moving average): mean=%.3f stddev=%.3f" % (self.mean, math.sqrt(self.variance))

    def verbose_string(self):
        if self.percent:
            return "(moving average): mean=%g%% stddev=%g" % (self.mean, math.sqrt(self.variance))
        else:
            return "(moving average): mean=%g stddev=%g" % (self.mean, math.sqrt(self.variance))
