#!/usr/bin/env python

import argparse
import sys
import os
import logging
from src.granalyzer.genius.LyricsScraper import LyricsScraper
from src.granalyzer.Corpus import Corpus


class Granalyzer(object):
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)

        parser = argparse.ArgumentParser(
            description='',
            usage='''granalyzer <command> [<args>]

The most commonly used commands are:
   generate     Generate lyrics corpus
   analyze      Analyze
''')
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        getattr(self, args.command)()

    def generate(self):
        parser = argparse.ArgumentParser(
            description='Generate corpus')
        parser.add_argument(
            'i',
            type=str,
            help='Input file')
        args = parser.parse_args(sys.argv[2:])
        file_path = os.path.abspath(args.i)

        scraper = LyricsScraper()
        scraper.generate_corpus(file_path=file_path)

    def analyze(self):
        parser = argparse.ArgumentParser(
            description='Generate corpus')
        parser.add_argument(
            'i',
            type=str,
            help='Input file')
        args = parser.parse_args(sys.argv[2:])
        file_path = os.path.abspath(args.i)

        corpus = Corpus(file_path)
        corpus.load_csv()
        corpus.prepare_corpus()

        corpus.compute_number_of_words()

        corpus.compute_lexical_richness()

        corpus.compute_wordcloud()

        corpus.compute_tf()


if __name__ == '__main__':
    Granalyzer()
