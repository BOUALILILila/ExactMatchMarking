#!/bin/bash
export field=description
export field_bm25=description
export collection=gov2
export out_suff=body
export k=1000
export plen=150
export overlap=75
export model=msmarco
export model_type=bert
export step=final

export DATA_DIR=/path/to/data/folder

STRATEGIES="base sim_pair"

echo "----------------------------"
echo "$collection - k=$k - ${field_bm25} - ${field} - ${model_type}"
echo "----------------------------"

for strategy in $STRATEGIES
do
    printf "\n+STRATEGY : $strategy\n"
    echo    "====================="
    
    if [ "${field}" == "${field_bm25}" ]; then
        export file_name=predictions_${collection}_${field}_doc_${out_suff}_${k}_p${plen}_o${overlap}_${strategy}_doc_level_marking_${model}-${step}_maxP.tsv
    else
        export file_name=predictions_${collection}_hybrid_doc_${out_suff}_${k}_p${plen}_o${overlap}_${strategy}_doc_level_marking_${model}-${step}_maxP.tsv
    fi

    printf "\nEvaluation of the BM25 run: \n"
    echo    "----------------------------"
    
    python /home/lila/data/data/Repositories/ExactMatchMarking/trec_eval.py \
                    --output_dir ./${collection}/trec_evaluations \
                    --qrels_path ${DATA_DIR}/qrels/qrels.${collection}.txt \
                    --preds_path ./${collection}/bm25/${collection}_run_${field_bm25}_${k}.tsv \
                    --per_query

    printf "\nEvaluation of the Re-Ranking run: \n"
    echo    "---------------------------------"
    
    python /home/lila/data/data/Repositories/ExactMatchMarking/trec_eval.py \
                    --output_dir ./${collection}/${model_type}/trec_evaluations \
                    --qrels_path ${DATA_DIR}/qrels/qrels.${collection}.txt \
                    --preds_path ./${collection}/${model_type}/${file_name} \
                    --per_query
    
    printf "\nLinear Combination using 5-fold cross-validation: \n"
    echo    "-------------------------------------------------"
    
    python /home/lila/data/data/Repositories/ExactMatchMarking/score_comb.py \
                    --output_dir ./${collection}/${model_type}/comb \
                    --doc_scores_path ./${collection}/bm25/${collection}_run_${field_bm25}_${k}.tsv \
                    --preds_path ./${collection}/${model_type}/${file_name} \
                    --qrels_path ${DATA_DIR}/qrels/qrels.${collection}.txt \
                    --folds_path ${DATA_DIR}/folds/${collection}-folds.json  

    printf "\nEvaluation of the linear comb: \n"
    echo    "--------------------------------"

    python /home/lila/data/data/Repositories/ExactMatchMarking/trec_eval.py \
                    --output_dir ./${collection}/${model_type}/trec_evaluations \
                    --qrels_path ${DATA_DIR}/qrels/qrels.${collection}.txt \
                    --preds_path ./${collection}/${model_type}/comb/comb_alpha_${file_name} \
                    --per_query
done