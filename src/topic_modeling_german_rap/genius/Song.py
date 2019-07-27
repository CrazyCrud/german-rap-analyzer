import re
import os


class Song:
    def __init__(self, **kwargs):
        self._artist = kwargs.get('artist')
        self._title = kwargs.get('title')
        self._year = kwargs.get('year')
        self._lyrics = kwargs.get('lyrics')

    @property
    def lyrics(self):
        return self._lyrics

    @property
    def year(self):
        return self._year

    @property
    def title(self):
        return self._title

    @property
    def artist(self):
        return self._artist

    def process_song_lyric(self):
        return re.sub(r'\[.*\]', '', self._lyrics)

    def save(self, output_dir, ext='txt'):
        return self._write_txt(output_dir, ext)

    def _write_txt(self, output_dir, ext):
        filename = self._sanitize_filename(output_dir, ext)
        if not os.path.isfile(filename):
            pass
            # TODO: save lyrics
        return filename

    def _sanitize_filename(self, output_dir, ext):
        filename = "{}_{}".format(self._artist.replace(" ", ""),
                                  self._title.replace(" ", "")).lower()
        filename = re.sub(r'[^a-zA-ZÄÖÜäöüß0-9\.\-_]', '', filename)
        filename = "{}.{}".format(filename, ext)

        return os.path.join(output_dir, filename)
