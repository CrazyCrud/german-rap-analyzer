import logging
import argparse
import gensim
import sys
import pickle
import csv
from preprocessing import prepare_text_for_lda
from visualize import visualize_lda
from genius.LyricsScraper import LyricsScraper
from genius.Song import Song

should_model_be_saved = False


class GermanRapAnalyzer:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Analyze german rap lyrics',
            usage='''granalyzer <command> [<args>]
            The most commonly used git commands are:
   generate_corpus     Download lyrics
   analyze_lyrics      Analyze lyrics
            ''')
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            logging.error('Unrecognized command')
            exit(1)
        getattr(self, args.command)()

    def generate_corpus(self):
        scraper = LyricsScraper()

        artist_names = scraper.read_artist_names()

        artists = scraper.get_artists(artist_names)
        scraper.save_lyrics(artists)

    def analyze_lyrics(self):
        corpus_data = self._get_csv_data()

        self._display_lyrics_length(corpus_data)

    def _get_csv_data(self):
        """
        Import the text data and the txt. file names in lists
        """
        corpus_data = []

        with open('corpus/corpus.csv', newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            for row in csv_reader:
                song = Song(row[0], row[1], row[2], row[3])

                tokens = prepare_text_for_lda(song.lyrics)
                song_data = {
                    'tokens': tokens,
                    'artist': song.artist,
                    'title': song.title,
                    'year': song.year
                }

                corpus_data.append(song_data)

        return corpus_data

    def _display_lyrics_length(self, corpus_data):
        pass
        # TODO

    def _display_most_frequent_words_grouped_by_year(self, corpus_data, lower_bound, upper_bound):
        pass
        # TODO: compute top 10 words before and after 2000
        # TODO: unigram and bisgram

    def _display_sentiment_score_grouped_by_year(self, corpus_data, lower_bound, upper_bound):
        pass
        # TODO

    def _display_sentiment_score_grouped_by_artist(self):
        pass
        # TODO

    def generate_lda_model(self, corpus, dictioanry, number_of_topics):
        lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=number_of_topics, id2word=dictionary, passes=15)
        if should_model_be_saved is True:
            lda_model.save('model{}.gensim'.format(NUM_TOPICS))
        return lda_model

    def find_topics(self, lda_model):
        topics = lda_model.print_topics(num_words=4)
        for topic in topics:
            print(topic)
        return lda_model

    def get_document_topics(self, lda_model, corpus):
        return lda_model.get_document_topics(corpus, minimum_probability=0.1)

    def create_similarity_matrix(self, corpus):
        tfidf_transform_tool = gensim.models.TfidfModel(corpus)
        corpus_tfidf = tfidf_transform_tool[corpus]

        index = gensim.similarities.MatrixSimilarity(tfidf_transform_tool[corpus])
        similarities = index[corpus_tfidf]

        """
        Print the similarity of one document to all others
        """
        print(list(enumerate(similarities)))

    def save_corpus(self, corpus):
        pickle.dump(corpus, open('corpus.pkl', 'wb'))
        dictionary.save('dictionary.gensim')

    def create_topic_models(self, text_data):
        """
        The dictionary contains unique tokens found in the document set
        """
        dictionary = gensim.corpora.Dictionary(text_data)

        """
        The tokens of the documents are stored in the variable corpus.
        They are stored in tuples where the id of the specific word is the first index
        and the second index represents the word count.
        corpus[10] = [(12, 3), (14, 1), ...] 
        """
        corpus = [dictionary.doc2bow(text) for text in text_data]

        """
        Save the corpus so it can be loaded to save some time
        """
        if should_model_be_saved is True:
            save_corpus(corpus)

        """
        Create the topic model
        """
        NUM_TOPICS = 5
        lda_model = generate_lda_model(corpus=corpus, dictioanry=dictionary, number_of_topics=NUM_TOPICS)

        """
        Find topics in the model
        """
        find_topics(lda_model=lda_model)

        """
        Get the topics of each document    
        """
        all_topics = get_document_topics(lda_model=lda_model, corpus=corpus)
        for index, doc_topics in enumerate(all_topics):
            print('{}'.format(text_labels[index]))
            print('Document topics: {}'.format(doc_topics))
            print('\n')

        """
        Look at the similarity of the documents
        """
        create_similarity_matrix(corpus=corpus)

        """
        Visualize the topic model
        """
        visualize_lda(lda=lda_model, corpus=corpus, dictionary=dictionary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    GermanRapAnalyzer()
