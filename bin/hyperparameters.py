"""
Module to update hyperparameters automatically.
"""

import configparser

def load():
    return config

class Hyperparameters:
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.readfp(open(config_file))

        data_vars = ["data_dir", "bin_dir", "run_dir", "run_name", "modelfile", "statefile", "training_sentences", "validation_sentences", "logfile", "verboselogfile"]

        training_vars = ["vocab_size", "curriculum_sizes", "batch_size", "init_embedding_range", "embedding_l1_penalty", "updates_per_normalize_embeddings", "validation_examples", "validation_logrank_noise_examples_percent", "embedding_size", "window_size", "hidden_size", "scale_init_weights_by", "activation_function", "learning_rate", "embedding_learning_rate", "validate_every", "validation_examples", "embedding_l1_penalty"]

        bool_vars = ["normalize_embeddings", "include_unknown_words"]

        for opt in data_vars:
            setattr(self, opt, config["data"][opt])

        for opt in training_vars:
            setattr(self, opt, config["training"][opt])

        for opt in bool_vars:
            setattr(self, opt, config.getboolean("training",opt))

        curriculum_sizes = int(*curriculum_sizes.split(","))
