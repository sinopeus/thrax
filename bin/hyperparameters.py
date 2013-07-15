
"""
Module to update hyperparameters automatically.
"""

from os.path import join
import sys
import configparser

config = configparser.ConfigParser()
with open('language-model.cfg') as configfile:
    config.readfp(configfile)
    
DATA_DIR = config.get("data", "data_dir")
RUN_NAME = config.get("data", "run_name")
VOCABULARY_SIZE = int(config.get("hyperparams", "vocab_size"))
INCLUDE_UNKNOWN_WORD = config.get("hyperparams", "include_unknown_words")

# HYPERPARAMETERS["TRAIN_SENTENCES"] = join(DATA_DIR, "%s.train.txt.gz" % RUN_NAME)
# HYPERPARAMETERS["ORIGINAL VALIDATION_SENTENCES"] = join(DATA_DIR, "%s.validation.txt.gz" % RUN_NAME)
# HYPERPARAMETERS["VALIDATION_SENTENCES"] = join(DATA_DIR, "%s.validation-%d.txt.gz" % (RUN_NAME, HYPERPARAMETERS["VALIDATION EXAMPLES"]))
# HYPERPARAMETERS["VOCABULARY"] = join(DATA_DIR, "vocabulary-%s-%d.txt.gz" % (RUN_NAME, VOCABULARY_SIZE))
# HYPERPARAMETERS["VOCABULARY_IDMAP_FILE"] = join(DATA_DIR, "idmap.%s-%d.include_unknown=%s.pkl.gz" % (RUN_NAME, VOCABULARY_SIZE, INCLUDE_UNKNOWN_WORD))
