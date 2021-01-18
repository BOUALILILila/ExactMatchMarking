import os
# import pyserini
# from pyserini.search import SimpleSearcher
from .utils import *
from .searcher import *


# split the lines with a ' ' or a '\t'
# rank starts at 0 or 1
# precision of the score (number of positions may differ)

def get_content_from_run_ids(
    run_path, col,
    data_path, anserini_path, index_path,
    use_doc_title, topic_field, K):


    collection = col.name().lower()
    topic_field = topic_field.lower()

    out_suff = 'title_body' if use_doc_title else 'body'
    doc_path = os.path.join(data_path, 'datasets', out_suff, f'{collection}_docs_{topic_field}_{K}.tsv')

    formated_run_path = os.path.join(data_path, 'datasets', out_suff, f'{collection}_run_{topic_field}_{K}.tsv')

    fqrel = os.path.join(data_path, 'qrels', 'qrels.' + collection + '.txt')
    qrels = load_qrels(fqrel)

    topic_field = topic_field.lower()
    topic_dir = os.path.join(data_path, 'topics')
    topic_path = os.path.join(topic_dir, topic_field , 'topics.' + collection + '.txt')
    qid2text = get_query(col, topic_path, topic_dir, topic_field)


    #searcher = SimpleSearcher(index_path)
    docsearch = Searcher(anserini_path)
    searcher = docsearch.build_searcher(index_path=index_path)

    dids = set()
    with open(doc_path,'w') as fdoc, open(formated_run_path,'w') as frun:
        with open(run_path) as run: 
            for line in run:
                qid, _, did, rank, score, _ = line.rstrip().split()
                if qid in qrels:
                    rel = qrels[qid][did] if did in qrels[qid] else 0
                    frun.write('{}\t{}\t{}\t{}\t{}\n'.format(qid, did,score, int(rank)+1, rel))
                    frun.flush()
                    if did not in dids:
                        dids.add(did)
                        doc = searcher.documentRaw(did)
                        if doc is None :
                            print(did, 'Not found')
                        else:
                            title, content = col.parse_doc(doc, use_doc_title=use_doc_title)
                                        
                            if use_doc_title:
                                if title in ('None'):
                                    title = ''
                            else:
                                title = ''
                            fdoc.write('{}\t{}\t{}\n'.format(did, title, content))
                            fdoc.flush()
        
    if not os.path.exists(topic_path):
        if not os.path.exists(topic_dir):
            os.makedirs(topic_dir)
        with open(topic_path, 'w', encoding='utf-8') as out_queries:
            for qid in qid2text:
                if qid in qrels:
                    out_queries.write('\t'.join([qid, qid2text[qid]])+'\n')
            