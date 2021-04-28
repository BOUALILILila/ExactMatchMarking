import pandas as pd
import numpy as np
import json
import argparse
import os
import pytrec_eval

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def main():
    
    parser = argparse.ArgumentParser(description='ScoreComb')

    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--doc_scores_path', type=str, required=True,
                            help='The path to the BM25 scores file .tsv file : q_id, doc_id, score, rank, judgement.')
    parser.add_argument('--preds_path', type=str, required=True,
                            help='The path to the maxP predictions file .tsv file : q_id, doc_id, score.')
    
    parser.add_argument('--qrels_path', type=str, required=True,
                            help='The path to the qrels file .txt file : q_id,_ ,doc_id, judgement.')
    parser.add_argument('--folds_path', type=str, required=True,
                            help='The path to the folds.')


    args, other = parser.parse_known_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    filename = os.path.join(args.output_dir, 'comb_alpha_' + args.preds_path.split('/')[-1])
    
    with open(args.folds_path) as f:
        folds = json.load(f)

    dfqrel = pd.read_csv(args.qrels_path, sep = ' ', header=None, dtype={0:str, 2:str})
    qrel = {str(k):{str(x[1]):int(x[2]) for x in v.values} for k, v in dfqrel[[0,2,3]].groupby(0)}
    dfqrel.columns = ['qid','_','did','rel']
        
    dfrun_A = pd.read_csv(args.doc_scores_path, sep = '\t', header=None, dtype={0:str, 1:str})
    dfrun_A.columns = ['qid','did','score','pos','rel']
        
    # maxP
    # dfrun_B = pd.read_csv(args.preds_path, sep = '\t', header=None, dtype={0:str, 1:str})
    # dfrun_B.columns = ['qid','did','score']
    
    # all
    dfrun_B = pd.read_csv(args.preds_path, sep = '\t', header=None, dtype={0:str, 1:str})
    dfrun_B.columns = ['qid','did','pass','logit_0','logit_1']
    dfrun_B = dfrun_B.sort_values(by=['qid','did','logit_1'], ascending=[True,True,False])
    dfrun_B = dfrun_B.drop_duplicates(['qid','did'], keep='first')
    logits = dfrun_B[['logit_0','logit_1']].values
    # #logsoftmax
    # dfrun_B['score'] = tf.nn.log_softmax(logits)[:,1].numpy()

    # #probabilitites softmax [0,1]
    dfrun_B['score'] = tf.nn.softmax(logits)[:,1].numpy()
    ## min-max norm to bm25 scores [0,1]
    scores = dfrun_A['score'].to_numpy().reshape(-1,1)
    scaler = MinMaxScaler()
    scaler.fit(scores)
    dfrun_A['score'] = scaler.transform(scores)
    
        
    dfrun = pd.merge(dfrun_A, dfrun_B, on=['qid','did'])

    dftest = [None]*len(folds)

    for itest in range(len(folds)):
        dftrain = dfrun[~dfrun['qid'].isin(folds[itest])]
        dftest[itest] = dfrun[dfrun['qid'].isin(folds[itest])]
        dftest[itest] = dftest[itest].copy()
        #foldqrel = {str(k):{x[1]:x[2] for x in v.values} for k, v in dftrain[['qid','did','rel']].groupby('qid')}
        foldqrel = {k:qrel[k] for k in dftrain.qid.values if k in qrel}
        foldevaluator = pytrec_eval.RelevanceEvaluator(foldqrel, {'map'})
        allcomb = [] 
        for di in np.arange(0.1,1.0,0.1):
            sdi = str(di)[:3]
            foldrun = {str(k):{x[1]:di*x[2]+(1-di)*x[3] for x in v.values} for k, v in dftrain[['qid','did','score_x','score_y']].groupby('qid')}
            perf = np.average([v['map'] for k,v in foldevaluator.evaluate(foldrun).items()])
            allcomb.append([perf,[di*x[0]+(1-di)*x[1] for x in dftest[itest][['score_x','score_y']].values]])
        bestcomb = np.argmax([x[0] for x in allcomb])
        print('> Fold:', itest)
        print("\tBest combination", bestcomb)
        print("\tPerformance",np.around(allcomb[bestcomb][0],4))
        print("\tAlpha",np.around(list(np.arange(0.1,1.0,0.1))[bestcomb],1))
        dftest[itest]['predict_merge'] = allcomb[bestcomb][1]
    dfAlphaComb = pd.concat([df for df in dftest])[['qid','did','predict_merge']]
    dfAlphaComb.to_csv(filename, sep='\t', index=None, header=None)
    
if __name__ == '__main__':
    main()

