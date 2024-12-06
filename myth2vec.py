from gensim.corpora.textcorpus import TextCorpus
from gensim.models import Word2Vec, FastText
import os
import logging
import numpy as np
from matplotlib import pyplot as plt
from radar_chart import radar_factory
import csv
import itertools as it
from collections import defaultdict
from scipy.stats import ttest_ind

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

""" Requires corpus as .txt files in ../MythFic_txt/ """

corpus_dir = '../MythFic_txt/'
assert corpus_dir.endswith('/')

metadata_file = '../fanfics_Greek_myth_metadata.csv'


characters = [
'hades',
'persephone',
'zeus',
'apollo',
'aphrodite',
'hera',
'ares',
'demeter',
'artemis',
'athena',
'achilles',
'hermes',
'patroclus',
'poseidon',
'dionysus',
'helen',
'hephaestus',
'icarus',
'odysseus',
'ariadne',
'hector',
'paris',
'hestia',
'cassandra',
'eros',
]

men = [
'hades',
'odysseus',
'zeus',
'hector',
'apollo',
'paris',
'ares',
'eros',
'achilles',
'hermes',
'patroclus',
'poseidon',
'dionysus',
'hephaestus',
'icarus'
]

women = [
'persephone',
'aphrodite',
'hera',
'demeter',
'artemis',
'athena',
'helen',
'ariadne',
'hestia',
'cassandra'
]




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

class MythCorpusGenreSplit(TextCorpus):
    """
    Same as above, but prepends genres to characters (e.g. fluff_hades)
    """
    max_chunk_size = 10000 # Break up lines longer than this, note that gensim word2vec silently truncates everything to 10k words

    def __init__(self, characters, *args, **kwargs):
        self.genre_char_count = defaultdict(int)
        self.characters = set(characters)
        self.fluff_ids = set()
        self.angst_ids = set()
        with open(metadata_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                tags = row['additional tags'].lower().split(', ')
                if 'fluff' in tags and not 'angst' in tags:
                    self.fluff_ids.add(row['work_id'])
                if 'angst' in tags and not 'fluff' in tags:
                    self.angst_ids.add(row['work_id'])

        super().__init__(*args, **kwargs)

    def getstream(self):
        num_texts = 0
        for i, filename in enumerate(os.listdir(self.input)):
            self.chars_in_current_doc = set() # only used for counting (in preprocess_genre_characters)

            ficid = os.path.splitext(filename)[0]
            genre = None
            if ficid in self.fluff_ids:
                genre = 'fluff'
                self.genre_char_count['fluff_total'] += 1
            elif ficid in self.angst_ids:
                genre = 'angst'
                self.genre_char_count['angst_total'] += 1

            if not filename.endswith('.txt'):
                continue
            filepath = os.path.join(corpus_dir, filename)
            with open(filepath, 'r') as f:
                for line in f.readlines():
                    if not line.strip():
                        continue
                    if line.count(' ') < self.max_chunk_size:
                        yield genre, line
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
                                yield genre, restline
                                break
                            else:
                                yield genre, restline[:chunkbound]
                            num_texts += 1

                            restline = restline[chunkbound+1:]
                            if restline.count(' ') < self.max_chunk_size:
                                print(restline.count(' '))
                                yield genre, restline
                                num_texts += 1
                                break

            for char in self.chars_in_current_doc:
                self.genre_char_count[char] += 1

        self.length = num_texts

    def preprocess_genre_characters(self, genre, line):
        """ caution, modifying input list (line), should not matter in practice though """
        for i, word in enumerate(line):
            if word in self.characters:
                self.chars_in_current_doc.add(f'all_{word}') # Count characters independent of genre
                if genre is not None:
                    line[i] = f'{genre}_{word}'
                    self.chars_in_current_doc.add(f'{genre}_{word}')
        return line

    def get_texts(self):
        """Generate documents from corpus.

        Yields
        ------
        list of str
            Document as sequence of tokens (+ lineno if self.metadata)

        """
        lines = self.getstream()
        if self.metadata:
            for lineno, (genre, line) in enumerate(lines):
                yield self.preprocess_genre_characters(genre, self.preprocess_text(line)), (lineno,)
        else:
            for genre, line in lines:
                yield self.preprocess_genre_characters(genre, self.preprocess_text(line))



def load_or_make_basic_corpus():
    corpuspickle_filename = f'{corpus_dir[:-1]}.corpuspickle'
    if os.path.exists(corpuspickle_filename):
        print("LOADING SAVED CORPUS PICKLE")
        corpus = MythCorpus.load(corpuspickle_filename)
    else:
        print("BUILDING NEW CORPUS AND SAVING WHEN DONE")
        corpus = MythCorpus(corpus_dir)
        corpus.save(corpuspickle_filename)
    return corpus

def load_or_make_genre_corpus():
    corpuspickle_filename = f'{corpus_dir[:-1]}.genre_corpuspickle'
    if os.path.exists(corpuspickle_filename):
        print("LOADING SAVED CORPUS PICKLE")
        corpus = MythCorpusGenreSplit.load(corpuspickle_filename)
    else:
        print("BUILDING NEW CORPUS AND SAVING WHEN DONE")
        corpus = MythCorpusGenreSplit(characters, corpus_dir)
        corpus.save(corpuspickle_filename)
    return corpus

class CorpusIter:
    """ Get the preprocessed sentences from the corpus in plaintext form, instead of
    (numerical) dictionary keys, saves a lot of hassle later. """
    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        return self.corpus.get_texts()

def load_or_make_model(corpus, pickle_suffix):
    """ Use another pickle_suffix when making a new model from another corpus """
    modelpickle_filename = f'{corpus_dir[:-1]}.{pickle_suffix}'
    if os.path.exists(modelpickle_filename):
        print("LOADING SAVED MODEL")
        model = Word2Vec.load(modelpickle_filename)
    else:
        print("BUILDING NEW MODEL AND SAVING WHEN DONE")
        # sg=1 means use skip-gram, like in the reference paper
        model = Word2Vec(sentences=CorpusIter(corpus), vector_size=300, sg=1, epochs=10)
        model.save(modelpickle_filename)
    return model

emotions = ['joy', 'fear', 'surprise', 'sadness', 'disgust', 'anger']

def emo(word, wv, sortit=False):
    similarity_tuples = [(wv.similarity(word, emotion), emotion) for emotion in emotions]
    if sortit:
        similarity_tuples.sort(reverse=True)
    return similarity_tuples

def emo_vector(word, wv):
    return np.array([wv.similarity(word, emotion) for emotion in emotions])

def plot_basic_radars(wv):
    for char in characters:
        N = len(emotions)
        values = emo_vector(char, wv)
        angles = np.linspace(0, 2*np.pi, N+1)
        # Initialise the spider plot
        ax = plt.subplot(111, polar=True)
         
        # Draw one axe per variable + add labels
        plt.xticks(angles[:-1], emotions, color='grey', size=8)
         
        # Draw ylabels
        ax.set_rlabel_position(0)
        ax.set_theta_zero_location("N")
        # plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
        plt.ylim(0,0.3)

        # Note: we need to repeat the first value to close the circular graph
        values_circ = list(values) + [values[0]]
        # Plot data
        ax.plot(angles, values_circ, linewidth=1, linestyle='solid')

        # Fill area
        ax.fill(angles, values_circ, 'b', alpha=0.1)
        ax.set_title(char, weight='bold', size='medium')

        os.makedirs('radarplots', exist_ok=True)
        plt.savefig(f'radarplots/{char}.png')
        plt.clf()

def plot_genre_radars(wv):
    for char in characters:
        N = len(emotions)
        try:
            values_angst = emo_vector(f'angst_{char}', wv)
            values_fluff = emo_vector(f'fluff_{char}', wv)
        except KeyError:
            print(f'{char} does not have both fluff and angst occurrences')
            continue

        angles = np.linspace(0, 2*np.pi, N+1)
        # Initialise the spider plot
        ax = plt.subplot(111, polar=True)
         
        # Draw one axe per variable + add labels
        plt.xticks(angles[:-1], emotions, color='grey', size=8)
         
        # Draw ylabels
        ax.set_rlabel_position(0)
        ax.set_theta_zero_location("N")
        # plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
        plt.ylim(0,0.3)

        # Note: we need to repeat the first value to close the circular graph
        values_angst_circ = list(values_angst) + [values_angst[0]]
        values_fluff_circ = list(values_fluff) + [values_fluff[0]]
        # Plot data
        ax.plot(angles, values_fluff_circ, linewidth=1, linestyle='solid', color='b', label='fluff')
        ax.plot(angles, values_angst_circ, linewidth=1, linestyle='solid', color='r', label='angst')

        # Fill area
        ax.fill(angles, values_fluff_circ, 'b', alpha=0.1)
        ax.fill(angles, values_angst_circ, 'r', alpha=0.1)

        ax.set_title(char, weight='bold', size='medium')
        ax.legend()

        os.makedirs('genreradarplots', exist_ok=True)
        plt.savefig(f'genreradarplots/{char}.png')
        plt.clf()

def plot_violin(wv):
    # male - female interspersed in the order of the emotions list
    emotion_data = []
    emotion_labels = []
    for emotion in emotions:
        emotion_data_man = [wv.similarity(emotion, man) for man in men]
        emotion_data_woman = [wv.similarity(emotion, woman) for woman in women]
        emotion_data.append(emotion_data_man)
        emotion_data.append(emotion_data_woman)
        emotion_labels += [f'male {emotion}', f'female {emotion}']

        print(f'{emotion} male vs. female t-test: {ttest_ind(emotion_data_man, emotion_data_woman)}')

    os.makedirs('violinplots', exist_ok=True)
    ax = plt.axes()
    violin = ax.violinplot(emotion_data, showmeans=True)
    ax.set_xticks(range(1, len(emotion_data)+1), emotion_labels, rotation=45, ha='right')

    for part, color in zip(violin['bodies'], it.cycle(['b','y'])):
        part.set_color(color)

    plt.tight_layout()
    plt.savefig(f'violinplots/separate.png')
    plt.clf()

def plot_violin_totals(wv):
    emotion_men = []
    emotion_women = []
    for man in men:
        emotion_men.append(np.mean([wv.similarity(emotion, man) for emotion in emotions]))
    for woman in women:
        emotion_women.append(np.mean([wv.similarity(emotion, woman) for emotion in emotions]))

    print(f'all emotions male vs. female t-test: {ttest_ind(emotion_men, emotion_women)}')

    os.makedirs('violinplots', exist_ok=True)
    ax = plt.axes()
    violin = ax.violinplot([emotion_men, emotion_women], showmeans=True)
    ax.set_xticks([1, 2], ['male total emotions', 'female total emotions'], rotation=45, ha='right')

    for part, color in zip(violin['bodies'], it.cycle(['b','y'])):
        part.set_color(color)

    plt.tight_layout()
    plt.savefig(f'violinplots/total.png')
    plt.clf()


def validate_vsm(wv):
    from gensim.test.utils import datapath
    # https://aclweb.org/aclwiki/WordSimilarity-353_Test_Collection_(State_of_the_art)
    # https://aclweb.org/aclwiki/Analogy_(State_of_the_art)
    print(wv.evaluate_word_pairs(datapath('wordsim353.tsv')))
    # print(wv.evaluate_word_analogies(datapath('questions-words.txt')))


if __name__ == '__main__':
    """ radar chart loosely adapted from https://python-graph-gallery.com/390-basic-radar-chart/ """

    corpus = load_or_make_basic_corpus()
    model = load_or_make_model(corpus, 'w2v_modelpickle')
    # Some analogies e.g. man : king :: woman : ?queen?
    """ https://radimrehurek.com/gensim/models/keyedvectors.html """
    print(model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man']))
    print(model.wv.similar_by_word('harry'))
    # plot_basic_radars(model.wv)
    plot_violin(model.wv)
    plot_violin_totals(model.wv)

    corpus_genre = load_or_make_genre_corpus()
    print('all', 'fluff', 'angst')
    for character in characters:
        print(corpus_genre.genre_char_count[f'all_{character}'], corpus_genre.genre_char_count[f'fluff_{character}'], corpus_genre.genre_char_count[f'angst_{character}'])
    model_genre = load_or_make_model(corpus_genre, 'w2v_genre_modelpickle')
    # plot_genre_radars(model_genre.wv)

    # validate_vsm(model.wv)
    # validate_vsm(model_genre.wv)


    # from plotumap import plot_model
    # plot_model(wv)


    # from plotemovectors import plot_emo
    # plot_emo([emo_vector(c) for c in characters], characters, emotions)
    # # lbls=['killing', 'death', 'fall', 'misery', 'hate', 'love', 'pleasant', 'beautiful', 'nice']
    # plot_emo([emo_vector(wrd) for wrd in lbls], lbls, lbls)
