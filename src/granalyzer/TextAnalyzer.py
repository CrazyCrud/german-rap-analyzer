from sklearn.feature_extraction.text import CountVectorizer


class TextAnalyzer:
    def __init__(self):
        pass

    def get_word_count(self, text):
        vectorizer = CountVectorizer()
    
        vectorizer.fit(text)
        print(vectorizer.vocabulary_)

        # vector = vectorizer.transform(text)
