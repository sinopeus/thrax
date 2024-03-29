"""
Module for storing hyperparameters.
"""

import configparser, numpy

class Hyperparameters:
    def __init__(self, config_file):
        self.config_file = config_file
        config = configparser.ConfigParser()
        config.readfp(open(config_file))

        data_vars = ["data_dir", "bin_dir", "run_dir", "run_name", "modelfile", "statefile", "training_sentences", "validation_sentences", "logfile", "verboselogfile", "dictionary"]

        training_vars_str = ["curriculum_sizes", "activation_function"]
        training_vars_int = ["vocab_size", "batch_size", "embedding_size", "window_size", "input_size","hidden_size", "output_size", "validate_every", "updates_per_normalize_embeddings", "rnd_seed"]
        training_vars_float = ["init_embedding_range", "validation_logrank_noise_examples_percent", "scale_init_weights_by", "learning_rate", "embedding_learning_rate"]
        training_vars_bool = ["normalize_embeddings", "include_unknown_words"]

        for opt in data_vars:
            setattr(self, opt, config["data"][opt])

        for opt in training_vars_str:
            setattr(self, opt, config.get("training", opt))

        for opt in training_vars_int:
            setattr(self, opt, config.getint("training", opt))

        for opt in training_vars_float:
            setattr(self, opt, numpy.float32(config.getfloat("training", opt)))

        for opt in training_vars_bool:
            setattr(self, opt, config.getboolean("training",opt))

        self.curriculum_sizes = [int(i) for i in self.curriculum_sizes.split(",")]
