
import os
import sys
from gensim.models import KeyedVectors

fname = "model.bin"

w2v = KeyedVectors.load_word2vec_format(fname, binary=True)
import pdb; pdb.set_trace()










