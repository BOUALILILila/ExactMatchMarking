import logging
import collections
import tensorflow as tf
import os , time
import spacy as sp

from .processor_utils import DataProcessor
from .data_utils import strip_html_xml_tags, clean_text
from .marker_utils import get_marker

logger = logging.getLogger(__name__)


# util functions for msmarco passage dataset
""" the collection tsv file must be preprocessed to fill the NaN titles with a '.' 
    documents with NaN body text have to be discarded """
def convert_eval_dataset(output_folder, 
                        qrels_path, 
                        queries_path, 
                        run_path, 
                        collection_path, 
                        set_name,
                        num_eval_docs,
                        sentence_level=True):
    print('Begin...')
    if not os.path.exists(output_folder):
            os.mkdir(output_folder)

    qrels = _load_qrels(qrels_path, set_name)

    queries = _load_queries(path=queries_path)
    run = _load_run(path=run_path)
    data = _merge(qrels=qrels, run=run, queries=queries)

    print('Loading Collection...')
    collection = _load_collection(collection_path)

    print('Converting to TFRecord...')
    _convert_dataset(data, collection, set_name, num_eval_docs, output_folder)

    print('Done!')

def _convert_dataset(data, 
                        collection, 
                        set_name, 
                        num_eval_docs, 
                        output_folder,
                        sentence_level = True):

    output_path = output_folder + f'/run_{set_name}_doc.tsv'
    start_time = time.time()
    random_title = list(collection.keys())[0]
    if sentence_level:
        sent_writer = open(output_folder + f'/run_{set_name}_sentence.tsv', 'w')
        nlp = sp.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
        nlp.max_length = 2100000
        nlp.add_pipe(nlp.create_pipe('sentencizer'))

    with open(output_path, 'w') as doc_writer:
        for idx, query_id in enumerate(data):
                query, qrels, doc_titles = data[query_id]

                clean_query = clean_text(query)

                doc_titles = doc_titles[:num_eval_docs]

                # Add fake docs so we always have max_docs per query.
                doc_titles += max(0, num_eval_docs - len(doc_titles)) * [random_title]

                labels = [
                    1 if doc_title in qrels else 0 
                    for doc_title in doc_titles
                ]

                len_gt_query = len(qrels)

                for label, doc_title in zip(labels, doc_titles):
                    title, doc = collection[doc_title]
                    _doc = strip_html_xml_tags(doc)
                    clean_doc = clean_text(_doc)
                    clean_title = clean_text(title)
                    if sentence_level:
                        d= nlp(clean_doc)
                        for i, sentence in enumerate(d.sents):
                            passage = sentence.string.strip()
                            doc_id = f'{doc_title}_{i}'
                            sent_writer.write("\t".join((query_id, doc_id, clean_query, title, passage, str(label), str(len_gt_query))) + "\n")
                    
                    doc_writer.write("\t".join((query_id, doc_title, clean_query, title, clean_doc, str(label), str(len_gt_query))) + "\n")

                if idx % 10 == 0:
                    print('wrote {} of {} queries'.format(idx, len(data)))
                    time_passed = time.time() - start_time
                    est_hours = (len(data) - idx) * time_passed / (max(1.0, idx) * 3600)
                    print('estimated total hours to save: {}'.format(est_hours))
    if sentence_level:
        sent_writer.close()


def _load_qrels(path, set_name):
    """Loads qrels into a dict of key: query_id, value: list of relevant doc ids."""
    qrels = collections.defaultdict(set)

    relevance_threshold = 1

    with open(path) as f:
        for i, line in enumerate(f):
                query_id, _, doc_id, relevance = line.rstrip().split()
                if int(relevance) >= relevance_threshold:
                    qrels[query_id].add(doc_id)
                if i % 1000 == 0:
                    print('Loading qrels {}'.format(i))
    return qrels


def _load_queries(path):
    """Loads queries into a dict of key: query_id, value: query text."""
    queries = {}
    with open(path) as f:
        for i, line in enumerate(f):
                query_id, query = line.rstrip().split('\t')
                queries[query_id] = query
                if i % 10 == 0:
                    print('Loading queries {}'.format(i))
    return queries


def _load_run(path):
    """Loads run into a dict of key: query_id, value: list of candidate doc ids."""
    # We want to preserve the order of runs so we can pair the run file with the
    # TFRecord file.
    run = collections.OrderedDict()
    with open(path) as f:
        for i, line in enumerate(f):
                query_id, doc_title, rank,_ = line.split('\t')
                if query_id not in run:
                    run[query_id] = []
                run[query_id].append((doc_title, int(rank)))
                if i % 1000 == 0:
                    print('Loading run {}'.format(i))
    # Sort candidate docs by rank.
    sorted_run = collections.OrderedDict()
    for query_id, doc_titles_ranks in run.items():
            sorted(doc_titles_ranks, key=lambda x: x[1])
            doc_titles = [doc_titles for doc_titles, _ in doc_titles_ranks]
            sorted_run[query_id] = doc_titles

    return sorted_run


def _merge(qrels, run, queries):
    """Merge qrels and runs into a single dict of key: query, 
        value: tuple(relevant_doc_ids, candidate_doc_ids)"""
    data = collections.OrderedDict()
    for query_id, candidate_doc_ids in run.items():
            query = queries[query_id]
            relevant_doc_ids = set()
            if qrels:
                relevant_doc_ids = qrels[query_id]
            data[query_id] = (query, relevant_doc_ids, candidate_doc_ids)
    return data


def _load_collection(path):
    """Loads tsv collection into a dict of key: doc id, value: doc text."""
    collection = {}
    with open(path) as f:
        for i, line in enumerate(f):
                doc_id, url, doc_title, doc_text = line.rstrip().split('\t')
                collection[doc_id] = (doc_title, doc_text.replace('\n', ' '))
                if i % 10000 == 0:
                    print('Loading collection, doc {}'.format(i))
    return collection


class MsMarcoDocumentProcessor(DataProcessor):

    def __init__(self, 
                document_handle,
                marker,
                ):
        super().__init__()
        self.handle = document_handle
        self.marker = marker
    
    def get_eval_dataset (self, data_path, batch_size, num_skip=0):
        return self.handle.get_eval_dataset(data_path, batch_size, num_skip)

    def prepare_inference_dataset( self,
                             data_path, 
                             output_dir,
                             set_name,
                              ):
        tf_writer = tf.io.TFRecordWriter(f"{output_dir}/dataset_{set_name}.tf")
        #tsv_writer = open(f"{output_dir}/pairs_{set_name}.tsv", 'w')
        ids_writer = open(f"{output_dir}/query_pass_ids_{set_name}.tsv", 'w')
        i_ids = 0

        start_time = time.time()

        print('Counting number of examples...')
        num_lines = sum(1 for line in open(data_path, 'r'))
        print('{} examples found.'.format(num_lines))
        
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i % 1000 == 0:
                    time_passed = int(time.time() - start_time)
                    print('Processed training set, line {} of {} in {} sec'.format(
                        i, num_lines, time_passed))
                    hours_remaining = (num_lines - i) * time_passed / (max(1.0, i) * 3600)
                    print('Estimated hours remaining to write the training set: {}'.format(
                        hours_remaining))
                              
                qid, did, query, title ,doc, label, len_gt_query = line.rstrip().split('\t')

                m_query, m_title, m_doc = self.marker.mark(query, title, doc)

                # write tfrecord
                i_ids = self.handle.write_eval_example(tf_writer, ids_writer, i_ids, m_query, [(m_title,m_doc)], [int(label)], qid, [did], int(len_gt_query)) ## stride pass_len
                #tsv_writer.write(f"{qid}\t{m_query}\t{did}\t{m_title}\t{m_doc}\t{label}\t{len_gt_query}\n")
        tf_writer.close()
        #tsv_writer.close()
        ids_writer.close()

