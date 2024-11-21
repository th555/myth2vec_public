import gensim
import os

""" Requires corpus as .txt files in ../MythFic_txt/ """

corpus_dir = '../MythFic_txt/'

max_chunk_size = 10000 # Break up lines longer than this, note that gensim word2vec silently truncates everything to 10k words

for i, filename in enumerate(os.listdir(corpus_dir)):
    filepath = os.path.join(corpus_dir, filename)
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if line.strip():
                nwords = line.count(' ')
                print(nwords)
                if nwords > max_chunk_size:
                    print("BIG")
                    exit()