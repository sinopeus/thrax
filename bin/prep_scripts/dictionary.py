import re, sys
from collections import Counter
chars = re.compile('[,\?\!#&_`\.%·; <>]')

def tokenise(line):
  return chars.split(line.strip().lower().replace("-", ""))

freqtable = Counter()

corpus_file = open(sys.argv[1])
for sentence in corpus_file:
  freqtable.update(Counter(tokenise(sentence)))

dictionary_file= open(sys.argv[2], 'a+')
for k, v in freqtable.most_common():
    print(k, file=dictionary_file)
