"""
Module to update hyperparameters automatically.
"""

from os.path import join
import sys
import configparser

def load():
    config = configparser.ConfigParser()
    with open('language-model.cfg') as configfile:
        config.readfp(configfile)
    return config
