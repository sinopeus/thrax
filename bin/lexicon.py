import pickle, re, json
from collections import Counter

class Corpus:
  def __init__(self, file):
    self.text = open(file).read()
    self.tokens = []
    self.freqtable = {}

  def tokenize(self):
    for line in iter(self.text):
      sentence = []
      for word in re.split('[,\?\!#&_`\.%·; <>]', line.strip().replace("-", "")): sentence.append(word.lower())
      self.tokens.append(sentence)

  def freqs(self):
    self.tokenize()
    self.freqtable = Counter(self.tokens)

  def most_common(self, number):
    self.freqs()
    return self.freqtable.most_common(number).keys()

class Dictionary:
  def __init__(self, modelfile=None, size=None):
    if modelfile != None:
      self.word_hash = pickle.load(open(modelfile))
    else:
      self.word_hash = self.build(TRAINING_SENTENCES, size)

  # def random(self, corpus, size=None):
  #   words = corpus.freqs(size)
  #   for word in words:
  #     self.word_hash[word] = numpy.asarray((numpy.random.rand(1, embedding_size) - 0.5)* 2 * 0.01, dtype=floatX)
    
  def update(self, word, embedding):
    self.word_hash[word] = embedding

  def dump(self):
    pickle.dump(self.word_hash, open(MODEL))

  def json_dump(self):
    print(json.dumps(self.word_hash, ensure_ascii=False), file=JSON_DUMP)
    
  def json_convert_dump(self):
    print(json.dumps(self, ensure_ascii=False), file=JSON_DUMP)
