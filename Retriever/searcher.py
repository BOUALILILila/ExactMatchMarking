# -*- coding: latin-1 -*-
'''This code is inspired from Birch (https://github.com/castorini/birch/tree/master/src/utils/)'''
import os, sys
import json
import re
import jnius_config
import glob
from io import open


#anserini version 0.9.4

import threading
class ThreadSafeSingleton(type):
    _instances = {}
    _singleton_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._singleton_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(ThreadSafeSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Searcher(metaclass=ThreadSafeSingleton):
    def __init__(self, anserini_path):
        paths = glob.glob(os.path.join(anserini_path, 'target', 'anserini-*-fatjar.jar'))
        if not paths:
            raise Exception('No matching jar file for Anserini found in target')

        latest = max(paths, key=os.path.getctime)
        jnius_config.set_classpath(latest)

        from jnius import autoclass
        self.JString = autoclass('java.lang.String')
        self.JSearcher = autoclass('io.anserini.search.SimpleSearcher')
        self.qidx = 1
        self.didx = 1

    def reset_idx(self):
        self.qidx = 1
        self.didx = 1

    def build_searcher(self, k1=0.9, b=0.4, fb_terms=10, fb_docs=10, original_query_weight=0.5,
        index_path='index/lucene-index.robust04.pos+docvectors+rawdocs', rm3=False):
        searcher = self.JSearcher(self.JString(index_path))
        searcher.setBM25(k1, b) # set_bm25()
        if rm3:
            searcher.setRM3(fb_terms, fb_docs, original_query_weight, False)
        return searcher

    def search_document(self, searcher, qid2docid, qid2text, test_qids, output_fn, field, 
                        use_doc_title, collection, 
                        K=1000, topics=None, filter_exact_matches=False, use_contents=False):
        col_name = collection.name().lower()
        with open(f'{output_fn}_run_{field}_{K}.tsv', 'w', encoding='utf-8') as out, \
            open(f'{output_fn}_docs_{field}_{K}.tsv', 'w', encoding='utf-8') as out_docs:

            doc_ids = set()
            if 'robust' not in col_name:
                # Robust04 provides CV topics
                topics = qid2text
            for qid in topics:
                if qid in test_qids:
                    text = qid2text[qid]
                    text_tokens = text.split()
                    hits = searcher.search(self.JString(text), K)
                    for i in range(len(hits)):
                        sim = hits[i].score
                        docno = hits[i].docid
                                                
                        label = 1 if qid in qid2docid and docno in qid2docid[qid] else 0
                        out.write('{}\t{}\t{}\t{}\t{}\n'.format(qid, docno,round(float(sim), 11), i+1, label))
                        out.flush()

                        if docno not in doc_ids:
                            doc_ids.add(docno)
                            hit = hits[i].contents if use_contents else hits[i].raw
                            title, content = collection.parse_doc(hit, use_doc_title=use_doc_title)
                                
                            if use_doc_title:
                                if title in ('None'):
                                    title = ''
                            else:
                                title = ''
                            out_docs.write('{}\t{}\t{}\n'.format(docno, title, content))
                            out_docs.flush()

    