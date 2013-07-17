"""
Module to update hyperparameters automatically.
"""

import configparser

def load():
    config = configparser.ConfigParser()
    with open('language-model.cfg') as configfile:
        config.readfp(configfile)
    return config
