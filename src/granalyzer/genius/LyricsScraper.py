import logging
import os
import csv
from .config import access_token
from .Song import Song
import lyricsgenius


class LyricsScraper:
    def __init__(self):
        self._genius = lyricsgenius.Genius(access_token, timeout=30)

    def generate_corpus(self, file_path):
        artist_names = self._read_artist_names(file_path)
        artists = self._get_artists(artist_names)
        self._save_lyrics(artists)

    def _read_artist_names(self, file_path):
        input_file = file_path

        return_artist_names = []
        try:
            with open(input_file, 'r', encoding='utf8') as f:
                for line in f:
                    if line is not None and len(line) > 0 and line[0] != '#':
                        return_artist_names.append(line)

        except Exception as e:
            logging.error(e)

        return return_artist_names

    def _get_artists(self, artist_names):
        return_artists = []
        for artist_name in artist_names:
            logging.debug("Search for artist {}".format(artist_name))

            artist = self._genius.search_artist(artist_name, max_songs=2)
            return_artists.append(artist)

        return return_artists

    def _save_lyrics(self, artists):
        script_dir = os.path.dirname(__file__)
        lyrics_dir = '../corpus'
        output_dir = os.path.join(script_dir, lyrics_dir)

        csv_file = os.path.join(output_dir, 'corpus.csv')
        with open(csv_file, 'w', newline='', encoding='utf8') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['artist', 'title', 'year', 'lyrics'])

            for artist in artists:
                songs = artist.songs
                for song in songs:
                    year = 0
                    if song.year is not None:
                        year = int(song.year[:4])
                    song = Song(artist=song.artist, title=song.title, year=year, lyrics=song.lyrics)

                    csv_writer.writerow([song.artist, song.title, song.year, song.clean_lyrics()])
