export k=1000
export field=title  # <title, description>
export topic_field=description  # <title, description> 
export run_name=hybrid # <title, description, hybrid> for title runs use title in both field and topic_field same goes for descritpion runs
export collection=robust04 # <robust04,gov2>
export num_eval_docs=1000
export plen=150
export overlap=75
export max_pass_per_doc=30

export out_suff=body  # without doc titles we use this setting for all our experiments feel free to test with titles (only for robust04 raw docs)

export DATA_DIR=/path/to/data/folder

python ./prep_data_plus.py \
			--collection ${collection} \
			--output_dir ${DATA_DIR}/pairs \
			--queries_path  ${DATA_DIR}/topics/${topic_field}/topics.${collection}.txt \
            --run_path ${DATA_DIR}/datasets/${out_suff}/${collection}_run_${field}_${k}.tsv \
            --collection_path ${DATA_DIR}/datasets/${out_suff}/${collection}_docs_${field}_${k}.tsv \ # replace by your corpus file if you want 
            --set_name ${collection}_${run_name}_doc_${out_suff}_${num_eval_docs}_p${plen}_o${overlap}\
			--num_eval_docs ${num_eval_docs} \
			--plen ${plen} \
			--overlap ${overlap} \
			--max_pass_per_doc ${max_pass_per_doc}
