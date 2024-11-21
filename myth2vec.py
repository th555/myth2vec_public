from gensim.corpora.textcorpus import TextCorpus
from gensim.models import Word2Vec
import os
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

""" Requires corpus as .txt files in ../MythFic_txt/ """

corpus_dir = '../MythFic_txt/'
assert corpus_dir.endswith('/')


class MythCorpus(TextCorpus):
    """
    Does some preprocessing by default:

    lower_to_unicode() - lowercase and convert to unicode (assumes utf8 encoding)
    deaccent() - deaccent (asciifolding)
    strip_multiple_whitespaces() - collapse multiple whitespaces into one
    simple_tokenize() - tokenize by splitting on whitespace
    remove_short_tokens() - remove words less than 3 characters long
    remove_stopword_tokens() - remove stopwords
    """
    max_chunk_size = 10000 # Break up lines longer than this, note that gensim word2vec silently truncates everything to 10k words

    def getstream(self):
        num_texts = 0
        for i, filename in enumerate(os.listdir(self.input)):
            if not filename.endswith('.txt'):
                continue
            filepath = os.path.join(corpus_dir, filename)
            with open(filepath, 'r') as f:
                for line in f.readlines():
                    if not line.strip():
                        continue
                    if line.count(' ') < self.max_chunk_size:
                        yield line
                        num_texts += 1
                    else:
                        print("SPLITTING")
                        restline = line
                        while restline:
                            # Chunk line to get around the (silent) 10k word limit in word2vec
                            chunkbound = 0
                            for _ in range(self.max_chunk_size):
                                chunkbound = restline.find(' ', chunkbound+1)
                            if chunkbound == -1:
                                yield restline
                                break
                            else:
                                yield restline[:chunkbound]
                            num_texts += 1

                            restline = restline[chunkbound+1:]
                            if restline.count(' ') < self.max_chunk_size:
                                print(restline.count(' '))
                                yield restline
                                num_texts += 1
                                break

        self.length = num_texts

corpuspickle_filename = f'{corpus_dir[:-1]}.corpuspickle'
if os.path.exists(corpuspickle_filename):
    print("LOADING SAVED CORPUS PICKLE")
    corpus = MythCorpus.load(corpuspickle_filename)
else:
    print("BUILDING NEW CORPUS AND SAVING WHEN DONE")
    corpus = MythCorpus(corpus_dir)
    corpus.save(corpuspickle_filename)

class CorpusIter:
    """ Get the preprocessed sentences from the corpus in plaintext form, instead of
    (numerical) dictionary keys, saves a lot of hassle later. """
    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        return corpus.get_texts()

modelpickle_filename = f'{corpus_dir[:-1]}.modelpickle'
if os.path.exists(modelpickle_filename):
    print("LOADING SAVED MODEL")
    model = Word2Vec.load(modelpickle_filename)
else:
    print("BUILDING NEW MODEL AND SAVING WHEN DONE")
    # sg=1 means use skip-gram, like in the reference paper
    model = Word2Vec(sentences=CorpusIter(corpus), vector_size=300, sg=1, epochs=10)
    model.save(modelpickle_filename)

wv = model.wv

# Some analogies e.g. man : king :: woman : ?queen?
print(wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man']))
print(wv.similar_by_word('harry'))

# import pdb; pdb.set_trace()

if __name__ == '__main__':
    from plotumap import plot_model
    plot_model(wv)
