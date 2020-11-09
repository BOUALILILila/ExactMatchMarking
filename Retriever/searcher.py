# -*- coding: latin-1 -*-
'''This code is inspired from Birch (https://github.com/castorini/birch/tree/master/src/utils/)'''
import os, sys
import json
import re
import jnius_config
import glob
from io import open


from .utils import parse_doc_from_index, clean_html

#anserini version 0.9.4

class Searcher:
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
                        use_doc_title, collection='robust04', 
                        K=1000, topics=None, filter_exact_matches=False):
        with open(f'{output_fn}_run_{field}.tsv', 'w', encoding='utf-8') as out, \
            open(f'{output_fn}_docs_{field}.tsv', 'w', encoding='utf-8') as out_docs:

            doc_ids = set()
            if 'robust' not in collection:
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
                            content = hits[i].raw
                            if collection == 'core17':
                                if use_doc_title:
                                    paraphs = content.split('\n', 1)
                                    title = paraphs[0]
                                    content=''
                                    if len(paraphs)>1 :
                                        content = paraphs[1]
                                
                            if collection == 'core18':
                                content_json = json.loads(content)
                                content = ''
                                title = ''
                                for each in content_json['contents']:
                                    if each is not None and 'content' in each.keys():
                                        if use_doc_title:
                                            if each['type'] =='title':
                                                title = each['content']
                                            elif each['type']  not in ('kicker','byline','date'):
                                                content += '{}\n'.format(each['content'])
                                        else:
                                            content += '{}\n'.format(each['content'])
                            
                            if collection == 'robust04':
                                title, content = parse_doc_from_index(content)
                            clean_content = clean_html(content, collection=collection)
                            clean_content = clean_content.replace('\n', ' ')
                            clean_content = clean_content.replace('\t', ' ')
                            if use_doc_title:
                                clean_title = clean_html(title, collection=collection)
                                if clean_title in ('None'):
                                    clean_title = ''
                            else:
                                clean_title = ''
                            out_docs.write('{}\t{}\t{}\n'.format(docno,clean_title, clean_content))
                            out_docs.flush()

    