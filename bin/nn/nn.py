from model.parameters import Parameters
import graph, copy, math, logging

class Network(Model):
    def __init__(self, hyperparameters, model, task, output_size):
        self.task = task # since we train different networks, we want to be able to distinguish them somehow
        self.hyperparameters = copy.deepcopy(hyperparameters)
        self.hyperparameters.output_size = output_size

        self.parameters = Parameters(self.hyperparameters)
        self.parameters.embeddings = model.parameters.embeddings
        self.parameters.hidden_weights = model.parameters.hidden_weights
        self.parameters.hidden_biases = model.parameters.hidden_biases
        
        self.trainer = Trainer()

        graph.hidden_weights = self.parameters.hidden_weights
        graph.hidden_biases = self.parameters.hidden_biases
        graph.output_weights = self.parameters.output_weights
        graph.output_biases = self.parameters.output_biases


    def train(self, inputs, correct_outputs, learning_rate, embedding_learning_rate):
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
            embedding_learning_rate = embedding_learning_rate * weights[0]
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
