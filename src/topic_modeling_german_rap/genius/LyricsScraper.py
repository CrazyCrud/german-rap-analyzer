import logging
import os
import re
import csv
from .config import access_token
from .Song import Song
import lyricsgenius


class LyricsScraper:
    def __init__(self):
        self._genius = lyricsgenius.Genius(access_token, timeout=30)
        self._songs = []

        script_dir = os.path.dirname(__file__)
        txt_dir = 'artist_names.txt'
        self._input_file = os.path.join(script_dir, txt_dir)

        lyrics_dir = '../corpus'
        self._output_dir = os.path.join(script_dir, lyrics_dir)

    def read_artist_names(self, file_path=None):
        if file_path is None:
            file_path = self._input_file

        return_artist_names = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line is not None and len(line) > 0 and line[0] != '#':
                        return_artist_names.append(line)

        except Exception as e:
            logging.error(e)

        return return_artist_names

    def get_artists(self, artist_names):
        return_artists = []
        for artist_name in artist_names:
            logging.debug("Search for artist {}".format(artist_name))

            artist = self._genius.search_artist(artist_name, max_songs=15)
            return_artists.append(artist)

        return return_artists

    def save_lyrics(self, artists):
        csv_file = os.path.join(self._output_dir, 'corpus.csv')
        with open(csv_file, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow('artist', 'title', 'year', 'lyrics')

            for artist in artists:
                songs = artist.songs
                for song in songs:
                    year = 0
                    if song.year is not None:
                        year = int(song.year[:4])
                    song = Song(artist=song.artist, title=song.title, year=year, lyrics=song.lyrics)

                    csv_writer.writerow([song.artist, song.title, song.year, song.process_song_lyric()])
