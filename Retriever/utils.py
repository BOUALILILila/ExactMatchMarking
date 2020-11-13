'''Some of this code is copied from Birch (https://github.com/castorini/birch/tree/master/src/utils/)'''
import sys
import re
import os
import ssl
from importlib import reload

reload(sys)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context



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

def clean_html(html, collection):
    """
    Copied from NLTK package.
    Remove HTML markup from the given string.
    :param html: the HTML string to be cleaned
    :type html: str
    :rtype: str
    """
    html = str(html)
    # First we remove inline JavaScript/CSS:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
    # Next we can remove the remaining tags:
    cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
    # Finally, we deal with whitespace
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    cleaned = re.sub(r"\t", " ", cleaned)
    if 'robust' not in collection.lower():
        cleaned = re.sub(r"\n", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()



