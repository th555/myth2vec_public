from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora.dictionary import Dictionary
from gensim.models import Word2Vec
import os
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

""" Requires corpus as .txt files in ../MythFic_txt/ """

corpus_dir = '../MythFic_mini/'
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

# Save/load dictionary because it is time consuming
pickle_filename = f'{corpus_dir[:-1]}.dictionarypickle'
if os.path.exists(pickle_filename):
    print("LOADING DICTIONARY FROM PICKLE")
    dictionary = Dictionary()
    dictionary.load(pickle_filename)
    corpus = MythCorpus(corpus_dir, dictionary=dictionary)
else:
    print("BUILDING NEW DICTIONARY AND SAVING WHEN DONE")
    corpus = MythCorpus(corpus_dir)
    corpus.dictionary.save(pickle_filename)


model = Word2Vec(sentences=corpus, vector_size=200)
