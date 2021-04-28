export k=1000
export field=title  # <title, description>
export topic_field=description  # <title, description> 
export run_name=hybrid # <title, description, hybrid> for title runs use title in both field and topic_field same goes for descritpion runs
export collection=robust04
export out_suff=body
export num_eval_docs=100
export num_train_docs=1000
export plen=150
export overlap=75
export sub_sample_train=0.1
export num_seg_perdoc=30

export DATA_DIR=/path/to/data/folder

for fold in {1..5}
do
    python ./prep_data_plus.py \
			--collection ${collection} \
			--output_dir ${DATA_DIR}/pairs/${collection}_${run_name}_doc_${out_suff}_p${plen}_o${overlap}/num-pass-perdoc-30/train_sample_${sub_sample_train}/fold-${fold}-train-${num_train_docs}-test-${num_eval_docs} \
			--folds_file_path ${DATA_DIR}/folds/${collection}-folds.json \
			--queries_path  ${DATA_DIR}/topics/${topic_field}/topics.${collection}.txt \
            --run_path ${DATA_DIR}/datasets/${out_suff}/${collection}_run_${field}_${k}.tsv \
            --collection_path ${DATA_DIR}/datasets/${out_suff}/${collection}_docs_${field}_${k}.tsv \
            --set_name ${collection}\
			--num_eval_docs_perquery ${num_eval_docs} \
			--num_train_docs_perquery ${num_train_docs} \
			--sub_sample_train ${sub_sample_train} \
			--fold ${fold} \
			--plen ${plen} \
			--overlap ${overlap} \
			--max_pass_per_doc ${num_seg_perdoc}
done