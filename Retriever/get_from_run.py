
import os
os.environ['PATH']=os.environ['PATH']+':/logiciels/jdk-13.0.1/bin'

import pyserini
from pyserini.search import SimpleSearcher
from .utils import *



def get_content_from_run_ids(
    run_path, col,
    data_path, index_path,
    use_doc_title, topic_field, K):

    collection = col.name().lower()
    topic_field = topic_field.lower()

    out_suff = 'title_body' if use_doc_title else 'body'
    doc_path = os.path.join(data_path, 'datasets', out_suff, f'{collection}_docs_{topic_field}_{K}.tsv')

    formated_run_path = os.path.join(data_path, 'datasets', out_suff, f'{collection}_run_{topic_field}_{K}.tsv')

    fqrel = os.path.join(data_path, 'qrels', 'qrels.' + collection + '.txt')
    qrels = load_qrels(fqrel)

    topic_field = topic_field.lower()
    topics_dir = os.path.join(data_path, 'topics')
    topic_path = os.path.join(topics_dir, topic_field , 'topics.' + collection + '.txt')
    qid2text = get_query(col, topic_path, topic_dir, topic_field)
    test_qids = get_test_qids(fqrel)


    searcher = SimpleSearcher(index_path)

    dids = set()
    with open(run_path) as run, open(doc_path,'w') as fdoc, open(formated_run_path,'w') as frun:
        for line in run:
            qid, _, did, rank, score, _ = line.rstrip().split('\t')
            frun.write('{}\t{}\t{}\t{}\t{}\n'.format(qid, did,score, rank+1, qrels[qid][did]))
            frun.flush()
            if did not in dids:
                dids.add(did)
                doc = searcher.doc(did)
                if doc is None :
                    print(did, 'Not found')
                else:
                    title, content = col.parse_doc(doc.raw(), use_doc_title=use_doc_title)
                                
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
                if qid in test_qids:
                    out_queries.write('\t'.join([qid, qid2text[qid]])+'\n')
            