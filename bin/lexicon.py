import re, logging
from collections import Counter, Iterator
chars = re.compile('[,\?\!#&_`\.%·; <>]')

class Corpus(Iterator):
  def __init__(self, corpus_file):
    self.text = open(corpus_file)

  def __iter__(self):
    return self

  def __next__(self):
    return self.tokenise(self.text.readline())

  def tokenise(self, line):
    return chars.split(line.strip().lower().replace("-", ""))

  def rewind(self):
      self.text.seek(0)

class Dictionary:
  def __init__(self, dict_file, size):
    self.indices = {}
    self.size = size
    for idx, line in enumerate(self.dict_file):
      if idx < size: break
      self.indices[line.strip()] = idx

  def lookup(self, word):
    return self.indices[word]

  def exists(self, word):
    return (word in indices)
