import logging
import argparse
import gensim
import sys
import pickle
import csv
import collections
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import generate_unigrams, generate_bigrams
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
   analyze_corpus      Analyze lyrics
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

    def analyze_corpus(self):
        corpus_data = self._get_csv_data()

        word_frequency_tuples = self._get_most_frequent_unigrams_by_year(corpus_data,
                                                                         lower_bound=1980,
                                                                         upper_bound=2000)

        self._display_most_frequent_unigrams_by_year(word_frequency_tuples)

        word_frequency_tuples = self._get_most_frequent_unigrams_by_year(corpus_data,
                                                                         lower_bound=2001,
                                                                         upper_bound=2019)

        self._display_most_frequent_unigrams_by_year(word_frequency_tuples)

        word_frequency_tuples = self._get_most_frequent_bigrams_by_year(corpus_data,
                                                                        lower_bound=1980,
                                                                        upper_bound=2000)

        self._display_most_frequent_bigrams_by_year(word_frequency_tuples)

        word_frequency_tuples = self._get_most_frequent_bigrams_by_year(corpus_data,
                                                                        lower_bound=2001,
                                                                        upper_bound=2019)

        self._display_most_frequent_bigrams_by_year(word_frequency_tuples)

    def _get_csv_data(self):
        """
        Import the csv data and split it columns
        """
        corpus_data = []

        with open('corpus/corpus.csv', newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            for row in csv_reader:
                song = Song(artist=row[0], title=row[1], year=int(row[2]), lyrics=row[3])

                unigrams = generate_unigrams(song.lyrics)
                bigrams = generate_bigrams(song.lyrics)
                song_data = {
                    'unigrams': unigrams,
                    'bigrams': bigrams,
                    'artist': song.artist,
                    'title': song.title,
                    'year': song.year
                }

                corpus_data.append(song_data)

        return corpus_data

    def _get_lyrics_length(self, corpus_data):
        pass
        # TODO

    def _get_most_frequent_unigrams_by_year(self, corpus_data, lower_bound, upper_bound):
        """
        Compute top 10 unigrams and bigrams between certain bounds0
        :param corpus_data:
        :param lower_bound:
        :param upper_bound:
        :return:
        """
        word_count = collections.defaultdict(int)

        number_of_filtered_items = 0

        corpus_data = filter(lambda corpus_item: lower_bound <= corpus_item['year'] <= upper_bound, corpus_data)
        for song_data in corpus_data:
            number_of_filtered_items += 1
            for token in song_data['unigrams']:
                word_count[token] += 1
        logging.debug("{} songs lie between {} and {}".format(number_of_filtered_items, lower_bound, upper_bound))

        return_frequency_tuples = []

        total_number_of_tokens = sum(word_count.values())
        word_count_sorted = sorted(word_count.items(), key=lambda kv: kv[1])
        word_count_sorted = collections.OrderedDict(word_count_sorted)

        for token, count in word_count_sorted.items():
            return_frequency_tuples.append((token, count / total_number_of_tokens))

        return_frequency_tuples = return_frequency_tuples[-10:]
        return_frequency_tuples = return_frequency_tuples[::-1]

        logging.debug("The frequncies from {} to {} are {}".format(lower_bound, upper_bound, return_frequency_tuples))

        return return_frequency_tuples

    def _get_most_frequent_bigrams_by_year(self, corpus_data, lower_bound, upper_bound):
        """
        Compute top 10 unigrams and bigrams between certain bounds0
        :param corpus_data:
        :param lower_bound:
        :param upper_bound:
        :return:
        """
        word_count = collections.defaultdict(int)

        number_of_filtered_items = 0

        corpus_data = filter(lambda corpus_item: lower_bound <= corpus_item['year'] <= upper_bound, corpus_data)
        for song_data in corpus_data:
            number_of_filtered_items += 1
            for token in song_data['bigrams']:
                word_count[token] += 1
        logging.debug("{} songs lie between {} and {}".format(number_of_filtered_items, lower_bound, upper_bound))

        return_frequency_tuples = []

        total_number_of_tokens = sum(word_count.values())
        word_count_sorted = sorted(word_count.items(), key=lambda kv: kv[1])
        word_count_sorted = collections.OrderedDict(word_count_sorted)

        for token, count in word_count_sorted.items():
            return_frequency_tuples.append((token, count / total_number_of_tokens))

        return_frequency_tuples = return_frequency_tuples[-10:]
        return_frequency_tuples = return_frequency_tuples[::-1]

        logging.debug("The frequncies from {} to {} are {}".format(lower_bound, upper_bound, return_frequency_tuples))

        return return_frequency_tuples

    def _display_most_frequent_unigrams_by_year(self, frequency_tuples):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        tokens = [frequency_tuple[0] for frequency_tuple in frequency_tuples]
        frequencies = [frequency_tuple[1] for frequency_tuple in frequency_tuples]

        ax.bar(tokens, frequencies)

        ax.set_xticklabels(tokens, rotation='horizontal', fontsize=6)

        ax.set_xlabel('Wörter')
        ax.set_ylabel('Relative Häufigkeit')

        plt.show()

    def _display_most_frequent_bigrams_by_year(self, frequency_tuples):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        tokens = [' '.join(frequency_tuple[0]) for frequency_tuple in frequency_tuples]
        frequencies = [frequency_tuple[1] for frequency_tuple in frequency_tuples]

        ax.bar(tokens, frequencies)

        ax.set_xticklabels(tokens, rotation='horizontal', fontsize=6)

        ax.set_xlabel('Wörter')
        ax.set_ylabel('Relative Häufigkeit')

        plt.show()

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
