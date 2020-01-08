import logging
import os
import re
import pandas as pd
import spacy
from spacy.lang.de.stop_words import STOP_WORDS
import nltk
from nltk.corpus import wordnet as wn

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

    def compute_lexical_richness(self):
        df = pd.DataFrame(columns=('artist', 'lexical_richness'))

        df_lyrics = self._df_songs.groupby('artist').agg({'lyrics': lambda x: list(x)})

        for index, row in df_lyrics.iterrows():
            logging.debug(row['artist'])

            a = len(set(row['lyrics']))
            b = len(row['lyrics'])
            df.loc[index] = (row['artist'], (a / float(b)) * 100)

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
