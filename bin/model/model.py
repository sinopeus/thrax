from model.parameters import Parameters
from model.graph import Graph
import math, logging, numpy

class Model:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.parameters = Parameters(self.hyperparameters)
        self.trainer = Trainer()
        self.graph = Graph(self.hyperparameters, self.parameters)

    def __getstate__(self):
        return (self.hyperparameters, self.parameters, self.trainer)

    def __setstate__(self, state):
        (self.hyperparameters, self.parameters, self.trainer) = state
        self.graph = Graph(self.hyperparameters, self.parameters)

    def embed(self, window):
        seq = [self.parameters.embeddings[word] for word in window]
        return numpy.dstack([numpy.resize(s, (1, s.size, 1)) for s in seq])

    def embeds(self, sequences):
        return numpy.vstack([self.embed(seq) for seq in sequences])

    def corrupt_example(self, e):
        import copy, random
        e = copy.deepcopy(e)
        pos = - self.hyperparameters.window_size // 2
        mid = e[pos]
        while e[pos] == mid: e[pos] = random.randint(0, self.hyperparameters.curriculum_size - 1)
        pr = 1. / self.hyperparameters.curriculum_size
        weight = 1. / pr
        return e, numpy.float32(weight)

    def corrupt_examples(self, correct_sequences):
        return zip(*[self.corrupt_example(e) for e in correct_sequences])

    def train(self, correct_sequences):
        noise_sequences, weights = self.corrupt_examples(correct_sequences)
        for w in weights: assert w == weights[0]
        learning_rate = self.hyperparameters.learning_rate

        r = self.graph.train(self.embeds(correct_sequences), self.embeds(noise_sequences), numpy.float32(learning_rate * weights[0]))

        correct_inputs_gradient, noise_inputs_gradient, losses, correct_scores, noise_scores = r

        to_normalize = set()
        for example in range(len(correct_sequences)):
            correct_sequence = correct_sequences[example]
            noise_sequence = noise_sequences[example]
            loss, correct_score, noise_score = losses[example], correct_scores[example], noise_scores[example]
            import pdb
            pdb.set_trace()

            correct_input_gradient = correct_inputs_gradient[example]
            noise_input_gradient = noise_inputs_gradient[example]

            # self.trainer.update(numpy.sum(loss), correct_score, noise_score)

            for w in weights: assert w == weights[0]
            embedding_learning_rate = self.hyperparameters.embedding_learning_rate * weights[0]
            if numpy.sum(loss) == 0:
                for di in correct_input_gradient + noise_input_gradient:
                    assert (di == 0).all()
            else:
                for (i, di) in zip(correct_sequence, correct_input_gradient.T):
                    self.parameters.embeddings[i] -= 1.0 * embedding_learning_rate * di
                    to_normalize.add(i)
                for (i, di) in zip(noise_sequence, noise_input_gradient.T):
                    self.parameters.embeddings[i] -= 1.0 * embedding_learning_rate * di
                    to_normalize.add(i)

            self.parameters.normalize(list(to_normalize))

    def predict(self, sequence):
        (score) = self.graph.predict(self.embed(sequence))
        return score

    def verbose_predict(self, sequence):
        (score, prehidden) = self.graph.verbose_predict(self.embed(sequence))
        return score, prehidden

    def validate(self, sequence):
        import copy
        corrupt_sequence = copy.copy(sequence)
        rank = 1
        correct_score = self.predict(sequence)
        mid = self.hyperparameters.window_size // 2

        for i in range(self.hyperparameters.curriculum_size - 1):
            if i == sequence[mid]: continue
            corrupt_sequence[mid] = i
            corrupt_score = self.predict(corrupt_sequence)
            rank += (correct_score <= corrupt_score)

        return rank

class Trainer:
    """
    We use a trainer to keep track of progress. This is, in effect, a
    wrapper object for all kinds of data related to training: average
    loss, average error, and a whole host of other variables.
    """
    def __init__(self):
        self.loss = MovingAverage()
        self.err = MovingAverage()
        self.lossnonzero = MovingAverage()
        self.squashloss = MovingAverage()
        self.correct_score = MovingAverage()
        self.noise_score = MovingAverage()
        self.cnt = 0

    def update(self, loss, correct_score, noise_score):
        self.loss.add(loss)
        self.err.add(int(correct_score <= noise_score))
        self.lossnonzero.add(int(loss > 0))
        squashloss = 1. / (1. + math.exp(-loss))
        self.squashloss.add(squashloss)
        self.correct_score.add(correct_score)
        self.noise_score.add(noise_score)
        self.cnt += 1

        if self.cnt % 10000 == 0: self.update_log()

    def update_log(self):
        logging.info(("After %d updates, pre-update train loss %s" % (self.cnt, self.loss.verbose_string())))
        logging.info(("After %d updates, pre-update train error %s" % (self.cnt, self.err.verbose_string())))
        logging.info(("After %d updates, pre-update train Pr(loss != 0) %s" % (self.cnt, self.lossnonzero.verbose_string())))
        logging.info(("After %d updates, pre-update train squash(loss) %s" % (self.cnt, self.squashloss.verbose_string())))
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
