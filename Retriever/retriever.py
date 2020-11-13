'''This code is inspired from Birch (https://github.com/castorini/birch/tree/master/src/utils/)'''
import os
import sys
import time
from .utils import *
from .searcher import *
from shutil import copyfileobj



def retrieve(col, args):

    collection = args.collection
    anserini_path = args.anserini_path
    data_path = args.data_path
    index_path = args.index_path
    out_suff = 'title_body' if args.use_doc_title else 'body'
    output_fn = os.path.join(args.data_path, 'datasets', out_suff , collection)

    fqrel = os.path.join(data_path, 'qrels', 'qrels.' + collection + '.txt')

    args.topic_field = args.topic_field.lower()
    
    topics_dir = os.path.join(data_path, 'topics')
    ftopic = os.path.join(topics_dir, args.topic_field , 'topics.' + collection + '.txt') 

    qid2docid = get_relevant_docids(fqrel)
    qid2text = get_query(col, ftopic, topics_dir, topic_field)
    #qid2title = get_query(os.path.join(data_path, 'topics', 'title', 'topics.' + collection + '.txt'))
    
    test_qids = get_test_qids(fqrel)
    docsearch = Searcher(anserini_path)

    print('Retrieving ...')
    print(f'Topic field: {args.topic_field}')
    print(f'Collection: \n\tName: {args.collection}\n\tUse document titles: {args.use_doc_title}')
    print(f'Retriever: \n\tK depth: {args.K}\n\tRM3: {args.rm3}')
    start_time = time.time()

    if collection == 'robust04':
        with open(os.path.join(data_path, 'folds', collection + '-folds.json')) as f:
            folds = json.load(f)
        # tuned params
        # params = ["0.9 0.5 47 9 0.30",
        #           "0.9 0.5 47 9 0.30",
        #           "0.9 0.5 47 9 0.30",
        #           "0.9 0.5 47 9 0.30",
        #           "0.9 0.5 26 8 0.30"]
        params = ["0.9 0.4 10 10 0.50",
                  "0.9 0.4 10 10 0.50",
                  "0.9 0.4 10 10 0.50",
                  "0.9 0.4 10 10 0.50",
                  "0.9 0.4 10 10 0.50"]
        folder_idx = 1
        for topics, param in zip(folds, params):
            # Extract each parameter
            k1, b, fb_terms, fb_docs, original_query_weight = map(float, param.strip().split())
            searcher = docsearch.build_searcher(k1=k1, b=b, fb_terms=fb_terms,
                                                fb_docs=fb_docs, original_query_weight=original_query_weight,
                                                index_path=index_path, rm3=args.rm3)
            docsearch.search_document(searcher, qid2docid, qid2text, test_qids,
                                      output_fn + str(folder_idx), args.topic_field,
                                      args.use_doc_title,
                                      col, args.K, topics)

            folder_idx += 1

        frun = os.path.join(args.data_path, 'datasets', out_suff, f'{collection}_run_{args.topic_field}_{args.K}.tsv')
        #Concat folds 
        with open(frun, 'w') as outfile:
            for infile in [f"{output_fn}{n}_run_{args.topic_field}_{args.K}.tsv" for n in range(1, 6)]:
                copyfileobj(open(infile), outfile)
        # Remove folds files
        for infile in [f"{output_fn}{n}_run_{args.topic_field}_{args.K}.tsv" for n in range(1, 6)]:
                os.remove(infile)

        fdocs= os.path.join(args.data_path, 'datasets', out_suff, f'{collection}_docs_{args.topic_field}_{args.K}.tsv')
        with open(fdocs, 'w') as outfile:
            for infile in [f'{output_fn}{n}_docs_{args.topic_field}_{args.K}.tsv' for n in range(1, 6)]:
                copyfileobj(open(infile), outfile)
        for infile in [f'{output_fn}{n}_docs_{args.topic_field}_{args.K}.tsv' for n in range(1, 6)]:
                os.remove(infile)
    
    # Core collections
    else:
        searcher = docsearch.build_searcher(k1=0.9, b=0.4, index_path=index_path, rm3=args.rm3)
        docsearch.search_document(searcher, qid2docid, qid2text, test_qids, output_fn,
                                 args.topic_field, args.use_doc_title, col, K=args.K)
                                 
    # Save the queries file if it is the first parsing of the topics
    if not os.path.exists(ftopic):
        topic_dir = os.path.join(topics_dir, args.topic_field)
        if not os.path.exists(topic_dir):
            os.makedirs(topic_dir)
        with open(ftopic, 'w', encoding='utf-8') as out_queries:
            for qid in qid2text:
                if qid in test_qids:
                    out_queries.write('\t'.join([qid, qid2text[qid]])+'\n')
    
    time_passed = int(time.time() - start_time)
    print(f'Retrieving Done. Durration: {time_passed} s')

