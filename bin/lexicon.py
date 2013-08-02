import re
from collections import Iterator
chars = re.compile('[,\?\!#&_`\.%·; <>]')

class Corpus(Iterator):
  def __init__(self, corpus_file, count=None):
    self.text = open(corpus_file)
    self.count = 0
    if count != None:
        self.__setstate__(count)

  def __iter__(self):
    return self

  def __next__(self):
    self.count += 1
    return self.tokenise(self.text.readline())

  def tokenise(self, line):
    return chars.split(line.strip().lower().replace("-", ""))

  def rewind(self):
    self.text.seek(0)

  def __getstate__(self):
    return (self.text.name, self.count)

  def __setstate__(self,count):
    while self.count < count:
        next(self)

class Dictionary(Iterator):
  def __init__(self, dict_file, size):
    self.indices = {}
    self.size = size
    self.dict_file = open(dict_file)
    self.build()

  def __iter__(self):
    return self

  def __next__(self):
    return self.dict_file.readline().strip()

  def build(self):
    idx = 0
    while idx < self.size:
      self.indices[next(self)] = idx
      idx +=1

  def enlarge(self, size):
    while self.size < size:
      self.indices[next(self)] = self.size
      self.size +=1

  def lookup(self, word):
    return self.indices[word]

  def exists(self, word):
    return (word in self.indices)

  def __getstate__(self):
      return (self.dict_file.name, self.size)
