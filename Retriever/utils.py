'''Some of this code is copied from Birch (https://github.com/castorini/birch/tree/master/src/utils/)'''
import sys
import re
import os
# import ssl
# from importlib import reload
import collections

# reload(sys)

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context



# nlp = spacy.load('en_core_web_lg')

def get_test_qids(fqrel):
    qids= set()
    with open(fqrel) as f:
        for line in f:
            qid, _, docid, score = line.replace('\n', '').strip().split()
            qids.add(qid)
    return qids


def get_query(col, ftopic, topics_dir, topic_field):
    qid2query={}
    if os.path.exists(ftopic):
        with open (ftopic) as f:
            for line in f:
                qid, topic = line.rstrip().split('\t')
                if qid not in qid2query:
                    qid2query[qid] = topic

    else: # need to create the file in the first execution
        filename = ftopic.split('/')[-1]
        path = os.path.join(topics_dir,filename)
        queries = col.parse_queries(path)
        if topic_field == 'title':
            qid2query = queries[topic_field]
        elif topic_field =='description':
            qid2query = queries['desc']
        # elif topic_field =='desc_key_words':
        #     for qid, desc in queries['desc'].items():
        #         qid2query[qid] = udel_converter.convert_query_to_udel(desc)
    
    return qid2query
        

# def build_udel_queries(ftopic):
#     qid2desc = get_description(ftopic,'robust04')
#     qid2title = get_query(ftopic, 'robust04')
#     qid2udel = {}
#     for qid in qid2title:
#         query = qid2title[qid].strip()
#         new = [w.text for w in nlp(query) if w.text not in stopwords]
#         desc = qid2desc[qid].strip()
#         new += [w.text for w in nlp(desc).ents]

#         qid2udel[qid] = ' '.join(new)
#     return qid2udel

def get_relevant_docids(fqrel):
    qid2docid = {}
    with open(fqrel) as f:
        for line in f:
            qid, _, docid, score = line.replace('\n', '').strip().split()
            if score != '0':
                if qid not in qid2docid:
                    qid2docid[qid] = set()
                qid2docid[qid].add(docid)
    return qid2docid

def load_qrels(fqrel):
    qrels = collections.OrderedDict()
    with open(fqrel) as f:
        for line in f:
            qid, _, docid, label = line.replace('\n', '').strip().split()
            if qid in qrels:
                qrels[qid].update({docid:label})
            else:
                qrels[qid] = {docid:label} 
    return qrels





