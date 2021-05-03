export k=1000
export field=title-description # <title, description, hybrid>  
export collection=robust04
export out_suff=body
export how=words
export strategy=base
export plen=150
export overlap=75
export max_seq_len=256
export tokenizer_type=electra # <electra, bert> to use the specific fast tokenizer

export DATA_DIR=/path/to/data/folder
export TOK_DIR=tokenizer name or path


python ./convert_dataset_to_tfrecord.py \
					--collection ${collection} \
                    --strategy ${strategy} \
                    --data_path ${DATA_DIR}/pairs/run_${collection}_${field}_doc_${out_suff}_${k}_p${plen}_o${overlap}_passages.tsv \
                    --output_dir ${DATA_DIR}/datasets/${tokenizer_type} \
                    --set_name ${collection}_${field}_doc_${out_suff}_${k}_p${plen}_o${overlap}_${strategy} \
					--tokenizer_name_path ${TOK_DIR} \
					--tokenizer_type ${tokenizer_type} \
					--max_seq_len ${max_seq_len}
