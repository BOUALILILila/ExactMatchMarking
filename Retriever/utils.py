'''This code is inspired from Birch (https://github.com/castorini/birch/tree/master/src/utils/)'''
import sys
import re
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


def get_title(ftopic, collection):
    qid2query = {}
    empty = False
    qid = -1
    with open(ftopic) as f:
        for line in f:
            if empty is True and int(qid) >= 0:
                qid2query[qid] = line.replace('\n', '').strip()
                empty = False
            # Get topic number
            tag = 'Number: '
            ind = line.find(tag)
            if ind >= 0:
                # Remove </num> from core18
                end_ind = -7 if collection == 'core18' else -1
                qid = str(int(line[ind + len(tag):end_ind]))
            # Get topic title
            tag = 'title'
            ind = line.find('<{}>'.format(tag))
            if ind >= 0:
                query = line[ind + len(tag) + 3:-1].strip()

                if len(query) == 0:
                    empty = True
                else:
                    qid2query[qid] = query.lower()

    return qid2query

def get_description(ftopic, collection):
    qid2query = {}
    empty = False
    qid = -1
    with open(ftopic) as f:
        for line in f:
            tag = 'desc'
            ind = line.find('</{}>'.format(tag))
            if ind >= 0:
                    empty = False  
            tag = 'narr'
            ind = line.find('<{}>'.format(tag))
            if ind >= 0:
                    empty = False
            if empty is True and int(qid) >= 0:

                qid2query[qid] += ' '+line.rstrip()
            # Get topic number
            tag = 'Number: '
            ind = line.find(tag)
            if ind >= 0:
                # Remove </num> from core18
                end_ind = -7 if collection == 'core18' else -1
                qid = str(int(line[ind + len(tag):end_ind]))
            # Get topic description
            tag = 'desc'
            ind = line.find('<{}>'.format(tag))
            if ind >= 0:
                    qid2query[qid] = ''
                    empty = True    
    return qid2query

def get_query(ftopic):
    qid2query={}
    with open (ftopic) as f:
      for line in f:
        qid, topic = line.rstrip().split('\t')
        if qid not in qid2query:
          qid2query[qid] = topic
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
    if 'robust' not in collection:
        cleaned = re.sub(r"\n", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def parse_doc_from_index(content):
    ls = content.split('\n')
    see_text = False
    see_title = False
    doc = ''
    title = ''
    for l in ls:
        l = l.replace('\n', '').strip()
        if '<TEXT>' in l:
            see_text = True
        elif '</TEXT>' in l:
            break
        elif see_text:
            if l == '<P>' or l == '</P>':
                continue
            doc += l + ' '
        elif '<HEADLINE>' in l:
            see_title =True
        elif '</HEADLINE>' in l :
            see_title = False
        elif see_title:
            if l == '<P>' or l == '</P>':
                continue
            title += l + ' '
    return title.strip(), doc.strip()

