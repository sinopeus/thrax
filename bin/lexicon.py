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

class Dictionary:
  def __init__(self, dict_file, size):
    self.indices = {}
    i = 0
    for word in iter(self.dict_file):
      if i > size
      self.indices[word.strip()] = i
      i += 1

  def lookup(self, word):
    return self.indices[word]

  def exists(self, word):
    return (word in indices)
