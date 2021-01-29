import pandas as pd
import numpy as np
import pytrec_eval
    

def trec_eval(preds: pd.DataFrame, qrels: pd.DataFrame, metrics: set):

    # read the qrels 
    qrel = {str(k):{str(x[1]):int(x[2]) for x in v.values} for k, v in qrels[['qid','did','label']].groupby('qid')}
    
    # create the evaluator
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)
    
    # create the run
    run = {str(k):{str(x[1]):x[2] for x in v.values} for k, v in \
                                         preds[['qid','did','pred_max']].groupby('qid')}


    results = evaluator.evaluate(run)

    eval_results = {}
    for metric in metrics:
        eval_results[metric] = np.around(np.average([v[metric] for k,v in results.items()]),4)

    return eval_results