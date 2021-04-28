# Enhancing BERT with Exact Match Signals
Deep neural models pretrained on auxiliary text tasks exemplified by BERT reported impressive gains in the ad hoc retrieval task. However, important cues of this task, such as *exact matching*, were rarely addressed in previous work, where relevance is formalized as a matching problem between two segments of text similarly to Natural Language Processing (NLP) tasks. In this work, we propose to explicitly mark the terms that exactly match between the query and the document in the input of BERT, assuming that it is capable of learning how to integrate the exact matching signal when estimating the relevance. Our simple yet effective approach reports improvements in the ranking accuracy for three ad hoc benchmark collections.

---

## Resources 
Fine-tuned models on the MS MARCO passage dataset:

| Model        | L / H    | Path |
|--------------|----------|------|
| ELECTRA-vanilla | 12 / 768 | [Download]() |
| ELECTRA-Sim-Pair    | 12 / 768 | [Download]() |
| ELECTRA-Pre-Pair    | 12 / 768 | [Download]() |
| ELECTRA-Pre-Doc    | 12 / 768 | [Download]() |
| BERT-vanilla    | 12 / 768 | [Download]() |
| BERT-Sim-Pair    | 12 / 768 | [Download]() |
| BERT-Pre-Pair    | 12 / 768 | [Download]() |
| BERT-Pre-Doc    | 12 / 768 | [Download]() |
---

## Set up the environment !
We use a retrieve-and-rerank architecture for our experiments where [Anserini](https://github.com/castorini/anserini) is used for the retriever stage. Our experiments were done under the 0.9.4 version of the library. Please follow the installation instructions on their Github repo. 

```
# Create virtual environment
pip install virtualenv
virtualenv -p python3.7 exctM_env
source exctM_env/bin/activate

# Install requirements
pip install -r requirements.txt
```
---
## Models fine-tuning
### Get the data ready !
* Get MsMarco passage train dataset.
```
DATA_DIR=./train_data
mkdir ${DATA_DIR}

wget https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz -P ${DATA_DIR}

tar -xvf ${DATA_DIR}/triples.train.small.tar.gz -C ${DATA_DIR}
```
* Generate the training pairs file. Very usefull when fine-tuning multiple models since this processing is time-consuming.
```
output_dir=<path/to/out>
dataset_path=${DATA_DIR}/triples.train.small.tsv
```

```
python ./prep_data_plus.py --collection msmarco \
                      --set train \
                      --dataset_path ${dataset_path} \
                      --output_dir ${output_dir} \
                      --set_name msmarco_train_small
```
* Apply a marking strategy to highlight the exact match signals and save the dataset to a TFRecord file.

**Note:** For the strategies pre_doc and pre_pair--that is our implementation of MarkedBERT[[1]](#1)--, use this script to add the precise markers to the vocabulary and initialize their embeddings:
```
python ./add_marker_tokens.py --save_dir <path/out/save/vocabulary/and/model> \
                              --tokenizer_name_path bert-base-uncased \# default or google/electra-base-discriminator
                              --name bert_base_uncased # the name of the extended vocabualry file (pre_tokenizer_${name}, pre_model_${name})
```

```
strategy=<base, sim_doc, sim_pair, pre_doc, pre_pair>
data_path = <path/to/pairs/file>
tokenizer_path_or_name = <path/to/tokenizer, tokenizer name in transformers> # default to 'bert-base-uncased' this need to be set to the path of the augmented tokenizer with precise marker tokens for the precise marking strategies.
```
```
python ./convert_dataset_to_tfrecord.py --collection msmarco \
                                        --set train \
                                        --strategy $strategy \
                                        --tokenizer_name_path ${tokenizer_path_or_name} \
                                        --data_path ${data_path} \
                                        --output_dir ${output_dir} \
                                        --set_name ${collection}_${strategy} # dataset name
```

### Fine-tune BERT
We use Google Free TPUs for our experiments. See our google colab notebooks under Notebooks/.
```
python ./run_model.py --output_dir ${output_dir} \ # save the model checkpoints on GCS by the TF2 checpoint manager
                        --do_train True \
                        --do_eval False \
                        --do_predict False \
                        --evaluate_during_training False \
                        --per_device_train_batch_size 16 \
                        --per_device_eval_batch_size 4 \
                        --learning_rate 3e-6 \
                        --weight_decay 0.01 \
                        --adam_epsilon 1e-6 \
                        --num_train_epochs -1 \
                        --max_steps 100000 \
                        --warmup_steps 10000 \
                        --logging_steps 1000 \
                        --save_steps 5000 \
                        --ckpt_name ${CHECKPOINT_NAME} \
                        --ckpt_dir ${CKPT_DIR} \ # path to save the checkpoint with HuggingFace trasnformers format 
                        --eval_all_checkpoints False \
                        --logging_dir ${LOG_DIR} \  # must be in GCS
                        --seed 42 \
                        --collection msmarco \
                        --marking_strategy ${STRATEGY} \
                        --train_data_dir ${DATA_DIR} \  # directory that contains the training dataset file(s)
                        --eval_data_dir  ${EVAL_DATA_DIR} \ # directory that contains the eval dataset file(s)
                        --train_set_name ${TRAIN_SET_NAME} \ # convention of naming: dataset_train_${TRAIN_SET_NAME}.tf
                        --eval_set_name ${EVAL_SET_NAME} \ # convention of naming: dataset_dev_${EVAL_SET_NAME}.tf
                        --test_set_name ${TEST_SET_NAME} \ # convention of naming: dataset_test_${TEST_SET_NAME}.tf
                        --out_suffix ${OUT_SUFFIX} \ # for the output file names: 
                        --eval_qrels_file ${EVAL_QRELS_FILE_NAME} \ # optional, for calculating evaluation measures
                        --test_qrels_file ${TEST_QRELS_FILE_NAME} \
                        --max_seq_length 512 \ 
                        --max_query_length 64 \
                        --model_name_or_path ${MODEL_NAME_OR_PATH} \ # bert-base-uncased or augmented version with precise tokens
```
---
## Inference 
### Index the collections !
We use [Anserini](https://github.com/castorini/anserini) for indexing our collections. Follow the quidlines:

- Robust04: https://github.com/castorini/anserini/blob/master/docs/regressions-robust04.md
- GOV2: https://github.com/castorini/anserini/blob/master/docs/regressions-gov2.md

Make sure to save the contents when indexing by setting the ```-storeContents``` flag if you want to use them for the document contents.

### Retrieve the top-K initial candidates list !
The data_path has the following structure:
```
data_path
|-- topics
|   |--title
|   |--description
|--folds
|--qrels
|--datasets
   |--body
   |--title_body
```
- Where topics contain the original topic files of the three collections ```topics.{collection}.txt``` that can be found in Anserini ressources. After the first execution using title|description queries a new file is created for each collection and topic field under the directory ```data_path/topics/{topic_field}/`topics.{collection}.txt``` with this format: 
```
{Qid}\t{title|description}
```
  > The functions for creating the topics files can be found under ```Retriever/utils```(get_title|get_description).
- Qrels contain the qrels files ```qrels.{collection}.txt``` that can be found in Anserini ressources.

```
collection=<robust04, gov2>
topic_field=<title, description>
anserini_path=<path/to/anserini/root>
index_path=<path/to/lucene/index>
data_path=<path/to/data/root>

python ./retrieve.py --collection ${collection} --topic_field ${topic_field} \
                               --data_path ${data_path} \
                               --anserini_path ${anserini_path} \
                               --index ${index_path} \  
                               --K 1000
```
You can choose various parameters for the retrieval: ```-rm3``` for RM3 expansion, ```-K``` for the depth, BM25 parameters can be set using ```-bm25_k1``` and ```-bm25_b```.

The document contents are obtained after parsing the raw documents (```-storeRaw``` needed when indexing) or directly from the contents by using the ```-use_contents``` flag of ```retrieve.py```, if they were saved during indexing using the ```-storeContents```.

You can check the ```Data/data_utils.py``` for different preprocssing possibilities that you can apply to a specific collection in ```Data/collections.py#parse_doc()```. 

For titles we use the default parameters, for descriptions we use the following setting:
- Robust04: b=0.6, k1=1.9
- GOV2: b=0.6 , k1=2.0

### Prepare the passages !
The retriever generates 3 files: 
* The run file: ```{qid}\t{did}\t{score}\t{rank}\t{judgement}```
* The corpus file: ```{did}\t{title}\t{body}``` , if ```-use_title_doc``` was not set, the title field would be empty('').

The corpus file can be obtained with other ways, check [PARADE](https://github.com/canjiali/PARADE#faq) for another option. If your corpus is in the ```{did}\t{body}``` format make sure to set the ```-from_raw_docs``` flag when preparing passages in the next step. 

Before marking we construct a unique data file containing the pairs of query and split passages for each collection. This file is only created once and used for all strategies.
```
collection=<robust04, gov2>
output_dir=<path/to/out>
run_path=<path/to/run/file>
queries_path=<path/to/queries/file>
collection_path=<path/to/corpus/file>
```

```
python ./prep_data_plus.py --collection $collection \
                      --output_dir ${output_dir} \
                      --queries_path ${queries_path} \
                      --run_path ${run_path} \
                      --collection_path ${collection_path} \
                      --set_name ${collection}_${query_field} \
                      --num_eval_docs 1000 \ 
                      --plen 150 \
                      --overlap 75 \
                      --max_pass_per_doc 30 # --from_raw_docs
```
Use a marking strategy to highlight the exact match signals of the document w.r.t the query in the pairs file generated above.  The query and document passages are marked then tokenized and finally saved in a TFRecord file.

```
strategy=<base, sim_doc, sim_pair, pre_doc, pre_pair>
data_path = <path/to/pairs/file>
tokenizer_path_or_name = <path/to/tokenizer, tokenizer name in transformers> # default to 'bert-base-uncased' this need to be set to the path of the augmented tokenizer with precise marker tokens for the precise marking strategies.
```
```
python ./convert_dataset_to_tfrecord.py --collection $collection \
                                        --strategy $strategy \
                                        --tokenizer_name_path ${tokenizer_path_or_name} \
                                        --max_seq_len 256 \
                                        --data_path ${data_path} \
                                        --output_dir ${output_dir} \
                                        --set_name ${collection}_${query_field}_${strategy} # dataset name
```
### Run the Model !
Check the notebook for the zero-shot trasnfer setting.

For more settings please check the scripts.

***
## References
<a id="1">[1]</a> 
Lila Boualili, Jose G. Moreno, and Mohand Boughanem. 2020. MarkedBERT: Integrating Traditional IR Cues in Pre-trained Language Models for Passage Retrieval. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '20). Association for Computing Machinery, New York, NY, USA, 1977–1980. DOI:https://doi.org/10.1145/3397271.3401194
