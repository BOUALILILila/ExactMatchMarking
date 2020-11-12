# Enhancing BERT with Exact Match Signals
This repository contains the code for reproducing the results of our paper: **Enhancing BERT based Ranking with Simple Exact Matching**.

## Introduction
Deep neural models pretrained on auxiliary text tasks exemplified by BERT reported impressive gains in the ad hoc retrieval task. However, important characteristics of this task, such as *exact matching*, were rarely addressed in previous work, where relevance is formalized as a matching problem between two segments of text similarly to Natural Language Processing (NLP) tasks. In this work, we propose to explicitly mark the terms that exactly match between the query and the document in the input of BERT, assuming that it is capable of learning how to integrate the exact matching signal when estimating the relevance. Our simple yet effective approach reports improvements in the ranking accuracy for three ad hoc benchmark collections.

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
* Generate the training pairs file.
```
output_dir=<path/to/out>
dataset_path=${DATA_DIR}/triples.train.small.tsv
```

```
python ./prep_data.py --collection msmarco \
                      --set train \
                      --dataset_path ${dataset_path} \
                      --output_dir ${output_dir} \
                      --set_name ${collection}_${query_field}
```
* Apply a marking strategy to highlight the exact match signals and save the dataset to a TFRecord file.

**Note:** For the strategies pre_doc and pre_pair--that is our implementation of MarkedBERT[[1]](#1)--, use this script to add the precise markers to the vocabulary and initialize their embeddings:
```
python ./add_marker_tokens.py --save_dir <path/out/save/vocabulary/and/model> \
                              --tokenizer_name_path bert-base-uncased \# default used in our experiments 
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
                                        --tokenizer_name_path ${tokenizer_path_or_name}
                                        --data_path ${data_path} \
                                        --output_dir ${output_dir} \
                                        --set_name ${collection}_${strategy} # dataset name
```

### Fine-tune BERT

---
## Inference on newswire collections
### Index the collections !
We use [Anserini](https://github.com/castorini/anserini) for indexing our collections. Follow the quidlines:

- Robust04: https://github.com/castorini/anserini/blob/master/docs/regressions-robust04.md
- Core17: https://github.com/castorini/anserini/blob/master/docs/regressions-core17.md
- Core18: https://github.com/castorini/anserini/blob/master/docs/regressions-core18.md

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
collection=<robust04, core17, core18>
topic_field=<title, description>
anserini_path=<path/to/anserini/root>
index_path=<path/to/lucene/index>
data_path=<path/to/data/root>
```
We use the default retrieve parameters: K=1000 for retrieval depth and use the RM3 exapansion. Can be changed via the parameters (K and rm3). the parameter ```use_title_doc``` indicates that the document title is considered.
```
python ./retrieve.py --collection ${collection} --topic_field ${topic_field} \
                               --data_path ${data_path} \
                               --anserini_path ${anserini_path} \
                               --index ${index_path} \  
                               --rm3 \
                               --use_title_doc 
```

### Prepare the passages !
The retriever generates 3 files: 
* The run file: ```{qid}\t{did}\t{score}\t{rank}\t{judgement}```
* The corpus file: ```{did}\t{title}\t{body}``` , if ```use_title_doc``` was set to False the title field would be empty('').

Before marking we construct a unique data file for each collection to be used for generating the datasets of the different strategies.
```
collection=<robust04, core17, core18>
output_dir=<path/to/out>
run_path=<path/to/run/file>
queries_path=<path/to/queries/file>
collection_path=<path/to/corpus/file>
```

```
python ./prep_data.py --collection $collection \
                      --output_dir ${output_dir} \
                      --queries_path ${queries_path} \
                      --run_path ${run_path} \
                      --collection_path ${collection_path} \
                      --set_name ${collection}_${query_field}
```
Use a marking strategy to highlight the exact match signals of the document w.r.t the query in the pairs file generated above. Each document is split to overlapping passages to overcome BERT's length limit. The query and document passages tokens are saved in a TFRecord file.

```
strategy=<base, sim_doc, sim_pair, pre_doc, pre_pair>
data_path = <path/to/pairs/file>
tokenizer_path_or_name = <path/to/tokenizer, tokenizer name in transformers> # default to 'bert-base-uncased' this need to be set to the path of the augmented tokenizer with precise marker tokens for the precise marking strategies.
```
```
python ./convert_dataset_to_tfrecord.py --collection $collection \
                                        --strategy $strategy \
                                        --tokenizer_name_path ${tokenizer_path_or_name} \
                                        --data_path ${data_path} \
                                        --output_dir ${output_dir} \
                                        --set_name ${collection}_${query_field}_${strategy} # dataset name
```
Check more arguments in the ```convert_dataset_to_tfrecord.py``` script like max_seq_length, passage length and stride. The default max lengths of the sequences is defaulted to : 
```
max_seq_len = 512
max_query_len = 64
max_title_len = 64
chunk_size = 384
stride = 192
```
For Robust04, since we don't use the title, the chunk size and stride can be augmented to 448 and 224 repsectively.

### Run the Model !


***
## References
<a id="1">[1]</a> 
Lila Boualili, Jose G. Moreno, and Mohand Boughanem. 2020. MarkedBERT: Integrating Traditional IR Cues in Pre-trained Language Models for Passage Retrieval. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '20). Association for Computing Machinery, New York, NY, USA, 1977–1980. DOI:https://doi.org/10.1145/3397271.3401194
