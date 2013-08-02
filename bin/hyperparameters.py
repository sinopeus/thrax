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

        training_vars_str = ["curriculum_sizes", "activation_function"]
        training_vars_int = ["batch_size", "embedding_size", "window_size", "hidden_size", "validate_every", "updates_per_normalize_embeddings"]
        training_vars_float = ["init_embedding_range", "embedding_l1_penalty", "validation_logrank_noise_examples_percent", "scale_init_weights_by", "learning_rate", "embedding_learning_rate"]
        training_vars_bool = ["normalize_embeddings", "include_unknown_words"]

        for opt in data_vars:
            setattr(self, opt, config["data"][opt])

        for opt in training_vars_str:
            setattr(self, opt, config.get("training", opt))
            
        for opt in training_vars_int:
            setattr(self, opt, config.getint("training", opt))

        for opt in training_vars_float:
            setattr(self, opt, config.getfloat("training", opt))

        for opt in training_vars_bool:
            setattr(self, opt, config.getboolean("training",opt))

        self.curriculum_sizes = [int(i) for i in self.curriculum_sizes.split(",")]
