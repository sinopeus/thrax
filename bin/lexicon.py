import pickle, re, json
from collections import Counter, Iterator

class Corpus(Iterator):
  def __init__(self, file):
    self.text = open(file)
    # self.sentences = []
    self.freqtable = Counter()

  def next(self):
    try:
      return tokenise(self.text.readline())
    except:
      raise StopIteration

  def tokenise(sentence):
    sentence = []
    for word in re.split('[,\?\!#&_`\.%·; <>]', line.strip().replace("-", "")): sentence.append(word.lower())
    self.sentences.append(sentence)
    return sentence

  def freqs(self):
    for line in iter(self):
      self.freqtable.update(Counter(tokenise(line)))
    self.text.seek(0, 0) # we do this once for each successive dictionary size and need to 'rewind' the corpus each time

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
    self.embeddings = {}

  # def random(self, corpus, size=None):
  #   words = corpus.freqs(size)
  #   for word in words:
  #     self.indices[word] = numpy.asarray((numpy.random.rand(1, embedding_size) - 0.5)* 2 * 0.01, dtype=floatX)
    
  def build(self, corpus):
    i = 0
    for word in corpus:
      self.indices[word] = i
      i += 1
  
  def update(self, embeddings):
    for word in self.indices.keys()
      self.embeddings[word] = embeddings[indices[word]]

  def lookup(self, word):
    return self.indices[word]
  
  def exists(self, word):
    return (word in indices)

  def serialise(self, filename):
    pickle.dump(self, filename)
  
  def json_dump(self, filename):
    print(json.dumps(self.embeddings, ensure_ascii=False), file=filename)
