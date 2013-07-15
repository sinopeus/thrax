import pickle, sys, tokenizer
from collections import Counter
from hyperparameters import *

class Corpus:
  def __init__(self, file):
    self.text = open(file).read()

  def tokenize(self):
    self.tokens = []

    for line in iter(self.text):
      sentence = []
      for word in re.split('[,\?\!#&_`\.%·; <>]', line.strip().replace("-", "")): sentence.append(word.lower())
      tokens.append(sentence)

  def freqs(self):
    self.tokenize()
    self.freqtable = Counter(tokens)

  def most_common(self, number):
    self.freqs()
    return freqtable.most_common(number).keys()

class Dictionary:
  def __init__(self, dictionary=None, number=None):
    if modelfile != None:
      self.word_hash = pickle.load(open(dictionary))
    else:
      self.word_hash = build(TRAINING_SENTENCES, number)

  def random(self, corpus, size=None):
    words = corpus.freqs(size)
    for word in words:
      self.word_hash[word] = numpy.asarray((numpy.random.rand(self.vocab_size, embedding_size) - 0.5)* 2 * 0.01, dtype=floatX)
    

  def update(self, word, embedding):
    word_hash[word] = embedding

  def dump(self):
    pickle.dump(self.word_hash, open(MODEL))

  def json_dump(self):
    dumpfile = open(location, "w")
    print(json.dumps(self.word_hash, ensure_ascii=false), file=JSON_DUMP)
    
  def json_convert_dump(self):
    lexicon = read_lex(location)
    dumpfile = open(location, "w")
    print(json.dumps(lexicon, ensure_ascii=false), file=JSON_DUMP)
