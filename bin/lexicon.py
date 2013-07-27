import re
from collections import Counter, Iterator

class Corpus(Iterator):
  def __init__(self, file):
    self.text = open(file)
    self.freqtable = Counter()

  def __iter__(self):
    return self

  def __next__(self):
    return self.tokenise(self.text.readline())

  def tokenise(self, line):
    return re.split('[,\?\!#&_`\.%·; <>]', line.strip().lower().replace("-", ""))

  def freqs(self):
    for sentence in self:
      self.freqtable.update(Counter(sentence))
    self.rewind()

  def rewind(self):
    self.text.seek(0, 0) # rather avoid missing out on part of the corpus ...

  def most_common(self, number):
    self.freqs()
    return self.freqtable.most_common(number).keys()

class Dictionary:
  def __init__(self, corpus=None, size=None):
    self.corpus = corpus
    self.indices = {}
    if corpus != None and size != None:
      self.build(corpus.most_common(size))
    self.size = size

  def build(self, freqs):
    i = 0
    for word in freqs:
      self.indices[word] = i
      i += 1

  def rebuild(self, size):
      self.build(self.corpus.most_common(size))

  def lookup(self, word):
    return self.indices[word]

  def exists(self, word):
    return (word in indices)
