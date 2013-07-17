import pickle, re, json
from collections import Counter 
from collections import Iterator

class Corpus(Iterator):
  def __init__(self, file):
    self.text = open(file)
    # self.sentences = []
    self.freqtable = Counter()

  def next(self):
    try:
      return self.text.readline()
    except:
      raise StopIteration

  # def tokenize(self):
  #   for line in iter(self.text.read()):
  #     sentence = []
  #     for word in re.split('[,\?\!#&_`\.%·; <>]', line.strip().replace("-", "")): sentence.append(word.lower())
  #     self.sentences.append(sentence)

  def freqs(self):
    for line in iter(self.text.read()):
      sentence = []
      for word in re.split('[,\?\!#&_`\.%·; <>]', line.strip().replace("-", "")): sentence.append(word.lower())
      self.freqtable.update(Counter(sentence))

  def most_common(self, number):
    self.freqs()
    self.text.seek(0, 0) # we do this once for each dictionary size and need to 'rewind' the corpus each time
    return self.freqtable.most_common(number).keys()

class Dictionary:
  def __init__(self, corpus=None, size=None):
    self.indices = {}
    if corpus != None and size != None:
      self.build(corpus.most_common(size))
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
  
  # def update(self, embeddings):
  #   for word in self.indices.keys()
  #     self.embeddings[word] = embeddings[indices[word]]

  def lookup(self, word):
    return self.indices[word]
  
  def exists(self, word):
    return (word in indices)

  def json_dump(self):
    print(json.dumps(self.indices, ensure_ascii=False), file=JSON_DUMP)
    
  def json_convert_dump(self):
    print(json.dumps(self, ensure_ascii=False), file=JSON_DUMP)
