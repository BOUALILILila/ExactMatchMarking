
import os
import argparse
import spacy

from .udel_stop_words import stopwords

class UdelConverter(object):
    def __init__(self, lg='en_core_web_lg'):
        self.nlp = spacy.load(lg)
        
    def convert_query_to_udel(self, query):
        q = query.lower().strip()
        new = [w.text for w in self.nlp(q) if (w.text not in stopwords) and (not w.is_punct) ]
        return ' '.join(new)