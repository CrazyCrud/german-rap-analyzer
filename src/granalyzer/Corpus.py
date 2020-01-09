import logging
import os
import re
import pandas as pd
import spacy
from spacy.lang.de.stop_words import STOP_WORDS
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import wordnet as wn

plt.rcParams['figure.figsize'] = (8, 6)

nlp = spacy.load('de')


class Corpus:
    def __init__(self, path):
        self._csv_path = os.path.abspath(path)

        self._df_songs = None

    def load_csv(self):
        """
        Import csv data
        """
        self._df_songs = pd.read_csv(self._csv_path, sep=';', encoding='utf8')

    def prepare_corpus(self):
        self._df_songs['lyrics'] = self._df_songs['lyrics'].map(Corpus.prepare_text)

        logging.debug(self._df_songs)

    def compute_number_of_words(self):
        df = pd.DataFrame(columns=('artist', 'words'))

        df_lyrics = self._df_songs.groupby('artist').agg({'lyrics': 'sum'})

        i = 0
        for index, row in df_lyrics.iterrows():
            a = len(row['lyrics'])
            df.loc[i] = (index, a)

            i += 1

        df.plot.bar(x='artist', y='words', title='Number of Words for each Artist')
        plt.show()

    def compute_lexical_richness(self):
        df = pd.DataFrame(columns=('artist', 'lexical_richness'))

        df_lyrics = self._df_songs.groupby('artist').agg({'lyrics': 'sum'})

        i = 0
        for index, row in df_lyrics.iterrows():
            a = len(set(row['lyrics']))
            b = len(row['lyrics'])
            df.loc[i] = (index, (a / float(b)) * 100)

            i += 1

        df.plot.bar(x='artist', y='lexical_richness', title='Lexical richness of each Artist')
        plt.show()

    def compute_wordcloud(self):
        df_lyrics = self._df_songs.groupby('artist').agg({'lyrics': 'sum'})

        script_dir = os.path.dirname(__file__)
        assets_dir = 'assets'
        output_dir = os.path.join(script_dir, assets_dir)

        for index, row in df_lyrics.iterrows():
            word_cloud = WordCloud(width=1000, height=500).generate(' '.join(row['lyrics']))

            # artist_name = re.sub(r'[^a-zA-ZÄÖÜäöüß0-9\.\-_]', '', index.lower())
            # word_cloud.to_file(f'{output_dir}{os.path.sep}{artist_name}.png')

            plt.title(index)
            plt.imshow(word_cloud.to_image(), interpolation="bilinear")
            plt.axis("off")
            plt.show()

    def compute_TF(self):
        df_lyrics = self._df_songs.groupby('artist').agg({'lyrics': 'sum'})

        for index, row in df_lyrics.iterrows():
            df = pd.DataFrame(columns=('word', 'tf'))

            words_count = len(row['lyrics'])

            for word, count in word_dict.items():
                df[word] = count / float(words_count)

    @staticmethod
    def prepare_text(text):
        text = text.lower()
        tokens = Corpus.tokenize(text)
        tokens = [Corpus.remove_special_characters(token) for token in tokens]
        tokens = [token for token in tokens if len(token) > 2]
        tokens = [token for token in tokens if token not in Corpus.get_stopwords()]

        return tokens

    @staticmethod
    def tokenize(text):
        return_tokens = []

        """
        Slice the text into separate tokens
        """
        tokens = nlp(text)

        """
        For each token check if it is a whitespace or non-white space (word).
        Then lemmatize it.
        """
        for token in tokens:
            if token.orth_.isspace():
                continue
            else:
                return_tokens.append(token.lemma_)
        return return_tokens

    @staticmethod
    def remove_special_characters(text):
        return re.sub('[^A-Za-z0-9äöüÄÖÜß]+', '', text)

    @staticmethod
    def get_stopwords():
        return set(STOP_WORDS)
