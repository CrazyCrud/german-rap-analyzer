import logging
import re
import csv
import spacy
from spacy.lang.de.stop_words import STOP_WORDS
import nltk
from nltk.corpus import wordnet as wn

nlp = spacy.load('de')


class CorpusProcessor:
    def __init__(self, path):
        self._csv_path = path
        self._csv_reader = None

    def load_csv(self):
        """
        Import csv data
        """
        self._csv_reader = csv.DictReader(open('corpus/corpus.csv', newline='', encoding='utf-8'))

    def prepare_corpus(self):
        for index, row in enumerate(self._csv_reader):
            self._csv_reader[index]['lyrics'] = self._prepare_text(row['lyrics'])

    def _prepare_text(self, text):
        text = text.lower()
        tokens = self._tokenize(text)
        tokens = [self._remove_special_characters(token) for token in tokens]
        tokens = [token for token in tokens if len(token) > 2]
        tokens = [token for token in tokens if token not in self._get_stopwords()]

        return tokens

    def _tokenize(self, text):
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

    def _remove_special_characters(self, text):
        return re.sub('[^A-Za-z0-9äöüÄÖÜß]+', '', text)

    def _get_stopwords(self):
        return set(STOP_WORDS)
