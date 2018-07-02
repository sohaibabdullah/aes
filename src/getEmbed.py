from .load_glove_embeddings import load_glove_embeddings
from ... import config
import os
filepath = config.dataset_path()

def loadGlove(embedding_dim):
    if embedding_dim == 50:
        file = os.path.join(filepath,'glove.6B.50d.txt')
        word2index, embedding_matrix = load_glove_embeddings(file, embedding_dim=50)
    if embedding_dim == 100:
        file = os.path.join(filepath,'glove.6B.100d.txt')
        word2index, embedding_matrix = load_glove_embeddings('glove.6B.100d.txt', embedding_dim=100)
    if embedding_dim == 200:
        file = os.path.join(filepath,'glove.6B.200d.txt')
        word2index, embedding_matrix = load_glove_embeddings('glove.6B.100d.txt', embedding_dim=200)
    if embedding_dim == 300:
        file = os.path.join(filepath,'glove.6B.300d.txt')
        word2index, embedding_matrix = load_glove_embeddings('glove.6B.100d.txt', embedding_dim=300)
    return word2index, embedding_matrix
