import numpy as np

from utils import timeit

GLOVE_DIM = 300
GLOVE_FILE = 'data/raw/glove.6B.%dd.txt' % GLOVE_DIM
VOCAB_FILE = 'data/raw/vocab_glove_complete.txt'


class GloveEmbedding(object):
    "Creates glove embedding object"

    def __init__(self, glove_file=GLOVE_FILE, glove_dim=GLOVE_DIM):
        with open(glove_file, encoding='utf-8') as fid:
            glove_txt = fid.readlines()
        glove_txt = [g.strip() for g in glove_txt]
        glove_vector = [g.split(' ') for g in glove_txt]
        glove_words = [g[0] for g in glove_vector]
        glove_dict = {w: i for i, w in enumerate(glove_words)}
        glove_vecs = [g[1:] for g in glove_vector]
        glove_array = np.zeros((glove_dim, len(glove_words)))
        for i, vec in enumerate(glove_vecs):
            glove_array[:,i] = np.array(vec)
        self.glove_array = glove_array
        self.glove_dict = glove_dict
        self.glove_words = glove_words
        self.glove_dim = glove_dim


class RecurrentEmbedding(object):
    "TODO"

    def __init__(self, glove_file=GLOVE_FILE, glove_dim=GLOVE_DIM,
                 vocab_file=VOCAB_FILE):
        self.glove_file = glove_file
        self.embedding = GloveEmbedding(self.glove_file, glove_dim)

        with open(vocab_file, encoding='utf-8') as fid:
            vocab = fid.readlines()
        vocab = [v.strip() for v in vocab]
        if '<unk>' in vocab:
            # don't have an <unk> vector.  Alternatively, could map to random
            # vector...
            vocab.remove('<unk>')

        self.vocab_dict = {}
        for i, word in enumerate(vocab):
            try:
                self.vocab_dict[word] = self.embedding.glove_array[
                    :, self.embedding.glove_dict[word]]
            except:
                print(f'{word} not in glove embedding')
