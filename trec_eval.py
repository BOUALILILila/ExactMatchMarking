import pandas as pd
import numpy as np
import argparse
import os
import json
import pytrec_eval
    

def trec_eval(preds_path: str, qrels_path: str, metrics: set):

    # read the qrels 
    dfqrel = pd.read_csv(qrels_path, sep = ' ', header=None, dtype={0:str, 2:str})
    qrel = {str(k):{str(x[1]):int(x[2]) for x in v.values} for k, v in dfqrel[[0,2,3]].groupby(0)}
    dfqrel.columns = ['qid','_','did','rel']
    
    # create the evaluator
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)

    # read the predictions
    dfrun = pd.read_csv(preds_path, sep = '\t', header=None,
                         dtype={0:str, 1:str}, usecols=[0,1,2])
    dfrun.columns = ['qid','did','pred']
    
    # create the run
    run = {str(k):{str(x[1]):x[2] for x in v.values} for k, v in \
                                         dfrun[['qid','did','pred']].groupby('qid')}


    results = evaluator.evaluate(run)

    return pd.DataFrame(results).T


def main():
    
    parser = argparse.ArgumentParser(description='TrecEval')

    parser.add_argument('--output_dir', type=str, required=False)

    parser.add_argument('--preds_path', type=str, required=True,
                            help='The path to the predictions file .tsv file : q_id, doc_id, score, ...')
    parser.add_argument('--qrels_path', type=str, required=True,
                            help='The path to the qrels file .txt file : q_id,_ ,doc_id, judgement.')

    parser.add_argument('--metrics', type=str, default='map ndcg_cut_20 P.20', 
                            help='Trec Eval metrics e.g. map, ndcg_cut, P. space separated if multiple')
    parser.add_argument('--per_query', type=bool, default=False, 
                            help='Get the metrics per query.')


    args, other = parser.parse_known_args()

    metrics = set(args.metrics.split())

    results = trec_eval(args.preds_path, args.qrels_path, metrics)

    if args.per_query:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        filename = os.path.join(args.output_dir, 'trec_eval_' + args.preds_path.split('/')[-1])
        results.to_csv(filename, sep='\t')
    
    for metric in results.columns:
        print(f'> {metric} = {np.around(np.average(results[metric]),4)}')

    
if __name__ == '__main__':
    main()
