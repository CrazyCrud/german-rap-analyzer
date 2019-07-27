import pyLDAvis.gensim


def visualize_lda(lda, corpus, dictionary):
    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.show(lda_display)
