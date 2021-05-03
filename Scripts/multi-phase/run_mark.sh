export k=1000
export field=title # <title, description, hybrid> 
export collection=robust04
export out_suff=body
export strategy=base
export plen=150
export overlap=75
export max_seq_len=256
export num_train_docs=1000
export num_eval_docs=100
export num_seg_perdoc=30
export train_prop=0.1
export tokenizer_type=electra


export DATA_DIR=/path/to/data/folder
export TOK_DIR=tokenizer name or path


export OUT_DIR=${DATA_DIR}/datasets/${tokenizer_type}/${collection}_${field}_doc_${out_suff}_p${plen}_o${overlap}_${max_seq_len}/num-pass-perdoc-${num_seg_perdoc}/train_sample_${train_prop}/fold-${fold}-train-${num_train_docs}-test-${num_eval_docs}
export FOLD_PATH=${DATA_DIR}/pairs/${collection}_${field}_doc_${out_suff}_p${plen}_o${overlap}/num-pass-perdoc-${num_seg_perdoc}/train_sample_${train_prop}/fold-${fold}-train-${num_train_docs}-test-${num_eval_docs}
export set_name=${collection}-fold-${SLURM_ARRAY_TASK_ID}-${strategy}

# test fold
python ./convert_dataset_to_tfrecord.py \
					--collection ${collection} \
                    --strategy ${strategy} \
                    --data_path ${FOLD_PATH}/run_test_${collection}_passages.tsv \
                    --output_dir ${OUT_DIR} \
                    --set_name ${set_name} \
					--tokenizer_name_path ${TOK_DIR} \
					--tokenizer_type ${tokenizer_type} \
					--max_seq_len ${max_seq_len}

# train folds		       
python ./convert_dataset_to_tfrecord.py \
                    --collection ${collection} \
					--set train \
                    --strategy ${strategy} \
                    --data_path ${FOLD_PATH}/run_train_${collection}_passages.tsv \
                    --output_dir ${OUT_DIR} \
                    --set_name ${set_name} \
                    --tokenizer_name_path ${TOK_DIR} \
					--tokenizer_type ${tokenizer_type} \
                    --max_seq_len ${max_seq_len}