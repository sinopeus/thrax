import pickle, sys, tokenizer
from collections import Counter
from hyperparameters import *

def read():
  try:
    lexicon = pickle.load(open(MODEL))
  except:
    lexicon = build(TRAINING_SENTENCES)

  return lexicon

def freqs(corpus):
  corpus = tokenizer.tokenize(corpus)
  merged_corpus = list(itertools.chain.from_iterable(corpus))
  return Counter(corpus)

def build(corpus, size=None):
  return dict(freqs(corpus).most_common(size)).keys()

word_hash = None
try:
  word_hash = read(MODEL)
except: pass

def write():
  pickle.dump(word_hash, open(MODEL))

def json_dump():
  dumpfile = open(location, "w")
  print(json.dumps(word_hash, ensure_ascii=false), file=JSON_DUMP)

def json_convert_dump():
  lexicon = read_lex(location)
  dumpfile = open(location, "w")
  print(json.dumps(lexicon, ensure_ascii=false), file=JSON_DUMP)

