import spacy
from spacy.lang.de.stop_words import STOP_WORDS

import nltk
from nltk.corpus import wordnet as wn

# nltk.download('wordnet')
# nltk.download('stopwords')

nlp = spacy.load('de')


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in get_stopwords()]
    # tokens = [get_lemma(token) for token in tokens]
    return tokens


def tokenize(text):
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
            lda_tokens.append(token_lemmatized.lower())
    return lda_tokens


def get_lemma(word):
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


def is_stopword(token):
    return token.is_stop


def get_stopwords():
    return set(STOP_WORDS)
