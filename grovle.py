import numpy as np

from utils import timeit

GrOVLE_FILE = './data/raw/grovle.txt'
GrOVLE_DIM = 300


class GrOVLEEmbedding(object):
    "Creates GrOVLE embedding object"

    def __init__(self, grovle_file=GrOVLE_FILE, grovle_dim=GrOVLE_DIM):
        with open(GrOVLE_FILE, encoding='utf-8') as fid:
            grovle_txt = fid.readlines()
        grovle_txt = [g.strip() for g in grovle_txt]
        grovle_vector = [g.split(' ') for g in grovle_txt]
        grovle_words = [g[0] for g in grovle_vector]
        grovle_vecs  = [g[1:] for g in grovle_vector]
        grovle_array = np.zeros((grovle_dim, len(grovle_words)))
        for i, vec in enumerate(grovle_vecs):
            grovle_array[:,i] = np.array(vec)
        grovle_dict  = {w: grovle_array[:,i] for i, w in enumerate(grovle_words)}
        self.vocab_dict = grovle_dict
        self.grovle_dim = grovle_dim
        self.missing_words = {}
    
    def __call__(self, query, max_words):
        len_query = min(len(query), max_words)
        feature = np.zeros((max_words, self.grovle_dim), dtype=np.float32)
        for i, word in enumerate(query[:len_query]):
            if word in list(self.vocab_dict.keys()):
                feature[i, :] = self.vocab_dict[word]  
            else:
                feature[i,:] = np.random.uniform(low=-1, high=1, 
                                size=(self.grovle_dim,))
                self.vocab_dict[word] = feature[i,:]
        return feature

if __name__ == '__main__':
    grovle = GrOVLEEmbedding()
    text = "Here is the sentence I want embeddings for."
    #Compute vectors
    feat = grovle(text)
    print(feat.shape)