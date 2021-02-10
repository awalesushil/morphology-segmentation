import re, grapheme
import numpy as np


class Utilities:
    def __init__(self):
        pass

    def filter(self, text):
        text = re.sub(r'\([^)]*\)', r'', text)
        text = re.sub(r'\[[^\]]*\]', r'', text)
        text = re.sub(r'<[^>]*>', r'', text)
        text = re.sub(r'[!।,\']', r'', text)
        text = re.sub(r'[०१२३४५६७८९]', r'', text)
        text = text.replace(u'\ufeff', '')
        text = text.replace(u'\xa0', u' ')
        text = re.sub(r'( )+', r' ', text)
        return text

    def tokenize(self, data_file):
        file        = open(data_file, "r")
        raw_text    = self.filter(file.read())
        return raw_text.split()

    def generate_splits(self, no_of_splits, tokens):

        splits = []

        for token in tokens[:100]:
            for s in range(no_of_splits):
                # Draw a sample from a Geometric Distribution
                split_point = np.random.geometric(p=0.5)
                stem = grapheme.slice(token, 0, split_point)
                stem = stem if (grapheme.length(stem) > 0) else '$'
                suffix = grapheme.slice(token, split_point)
                suffix = suffix if (grapheme.length(suffix) > 0) else '$'
                splits.append((stem, suffix))

        print('Total data:', len(splits))
        print('Data Sample \n', splits[:5])

        return splits