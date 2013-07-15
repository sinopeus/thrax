import re

"""
Return a list of tokenized sentences in list form. Both files and lists of strings are accepted as arguments.
"""
def tokenize(corpus):
    tokenized_corpus = []
    
    try: corpus = open(corpus)
    except : pass
    
    for line in iter(corpus):
        sentence = []
        for word in re.split('[,\?\!#&_`\.%·; <>]', line.strip().replace("-", "")): sentence.append(word.lower())
        tokenized_corpus.append(sentence)
        
    return tokenized_corpus
