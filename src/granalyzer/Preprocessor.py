import logging
import re
import spacy
from spacy.lang.de.stop_words import STOP_WORDS
import nltk
from nltk.corpus import wordnet as wn

# nltk.download('wordnet')
# nltk.download('stopwords')

nlp = spacy.load('de')


class Preprocessor:
    def __init__(self):
        pass

    def generate_unigrams(self, text):
        tokens = self._tokenize_to_unigrams(text)
        tokens = [token for token in tokens if len(token) > 2]
        tokens = [token for token in tokens if token not in self.get_stopwords()]
        # tokens = [get_lemma(token) for token in tokens]
        return tokens

    def generate_bigrams(self, text):
        tokens = self._tokenize_to_bigrams(text)
        tokens = [token for token in tokens if len(token[0]) > 2 and len(token[1]) > 2]
        tokens = [token for token in tokens if
                  token[0] not in self.get_stopwords() and token[1] not in self.get_stopwords()]
        # tokens = [get_lemma(token) for token in tokens]
        return tokens

    def _tokenize_to_unigrams(self, text):
        lda_tokens = []

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
                token_lemmatized = token.lemma_
                token_lemmatized = token_lemmatized.lower()
                token_cleared = re.sub('[^A-Za-z0-9äöüÄÖÜß]+', '', token_lemmatized)
                lda_tokens.append(token_cleared)
        return lda_tokens

    def _tokenize_to_bigrams(self, text):
        bigrams = list(nltk.bigrams(text.split()))
        bigrams = [
            (re.sub('[^A-Za-z0-9äöüÄÖÜß]+', '', bigram[0]).lower(),
             re.sub('[^A-Za-z0-9äöüÄÖÜß]+', '', bigram[1]).lower())
            for bigram
            in bigrams]
        return bigrams

    def get_lemma(self, word):
        """
        Create a lemmatized version of a word
        :param word:
        :return:
        """
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma

    def is_stopword(self, token):
        return token.is_stop

    def get_stopwords(self):
        return set(STOP_WORDS)
