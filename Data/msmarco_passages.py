import logging
import collections
import tensorflow as tf
import os, time

from .processor_utils import DataProcessor
from .data_utils import strip_html_xml_tags, clean_text
from .marker_utils import get_marker

from .convert_to_udel import UdelConverter

logger = logging.getLogger(__name__)

MINI_DEV = {'484694', '836399', '683975', '428803', '1035062', '723895', '267447', '325379', '582244', '148817', '44209', '1180950', '424238', '683835', '701002', '1076878', '289809', '161771', '807419', '530982', '600298', '33974', '673484', '1039805', '610697', '465983', '171424', '1143723', '811440', '230149', '23861', '96621', '266814', '48946', '906755', '1142254', '813639', '302427', '1183962', '889417', '252956', '245327', '822507', '627304', '835624', '1147010', '818560', '1054229', '598875', '725206', '811871', '454136', '47069', '390042', '982640', '1174500', '816213', '1011280', '368335', '674542', '839790', '270629', '777692', '906062', '543764', '829102', '417947', '318166', '84031', '45682', '1160562', '626816', '181315', '451331', '337653', '156190', '365221', '117722', '908661', '611484', '144656', '728947', '350999', '812153', '149680', '648435', '274580', '867810', '101999', '890661', '17316', '763438', '685333', '210018', '600923', '1143316', '445800', '951737', '1155651', '304696', '958626', '1043094', '798480', '548097', '828870', '241538', '337392', '594253', '1047678', '237264', '538851', '126690', '979598', '707766', '1160366', '123055', '499590', '866943', '18892', '93927', '456604', '560884', '370753', '424562', '912736', '155244', '797512', '584995', '540814', '200926', '286184', '905213', '380420', '81305', '749773', '850038', '942745', '68689', '823104', '723061', '107110', '951412', '1157093', '218549', '929871', '728549', '30937', '910837', '622378', '1150980', '806991', '247142', '55840', '37575', '99395', '231236', '409162', '629357', '1158250', '686443', '1017755', '1024864', '1185054', '1170117', '267344', '971695', '503706', '981588', '709783', '147180', '309550', '315643', '836817', '14509', '56157', '490796', '743569', '695967', '1169364', '113187', '293255', '859268', '782494', '381815', '865665', '791137', '105299', '737381', '479590', '1162915', '655989', '292309', '948017', '1183237', '542489', '933450', '782052', '45084', '377501', '708154'}

# util functions for msmarco passage dataset
def convert_eval_dataset(output_folder, 
                        qrels_path, 
                        queries_path, 
                        run_path, 
                        collection_path, 
                        set_name,
                        num_eval_docs):
    
    if not os.path.exists(output_folder):
            os.mkdir(output_folder)

    qrels = _load_qrels(qrels_path, set_name)

    queries = _load_queries(path=queries_path)
    run = _load_run(path=run_path)
    data = _merge(qrels=qrels, run=run, queries=queries)

    print('Loading Collection...')
    collection = _load_collection(collection_path)

    print('Converting to TFRecord...')
    _convert_dataset(data,collection, set_name, num_eval_docs, output_folder)

    print('Done!')

def _convert_dataset(data, 
                        collection, 
                        set_name, 
                        num_eval_docs, 
                        output_folder):

    output_path = output_folder + f'/run_{set_name}_full.tsv'
    start_time = time.time()
    random_title = list(collection.keys())[0]

    with open(output_path, 'w') as writer:
        for i, query_id in enumerate(data):
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
                    _doc = strip_html_xml_tags(collection[doc_title])
                    clean_doc = clean_text(_doc)

                    writer.write("\t".join((query_id, doc_title, clean_query, clean_doc, str(label), str(len_gt_query))) + "\n")

                if i % 1000 == 0:
                    print('wrote {} of {} queries'.format(i, len(data)))
                    time_passed = time.time() - start_time
                    est_hours = (len(data) - i) * time_passed / (max(1.0, i) * 3600)
                    print('estimated total hours to save: {}'.format(est_hours))


def _load_qrels(path, set_name):
    """Loads qrels into a dict of key: query_id, value: list of relevant doc ids."""
    qrels = collections.defaultdict(set)

    relevance_threshold = 2 if set_name=='test' else 1

    with open(path) as f:
        for i, line in enumerate(f):
                query_id, _, doc_id, relevance = line.rstrip().split('\t')
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
                if i % 1000 == 0:
                    print('Loading queries {}'.format(i))
    return queries


def _load_run(path):
    """Loads run into a dict of key: query_id, value: list of candidate doc ids."""
    # We want to preserve the order of runs so we can pair the run file with the
    # TFRecord file.
    run = collections.OrderedDict()
    with open(path) as f:
        for i, line in enumerate(f):
                query_id, doc_title, rank = line.split('\t')
                if query_id not in run:
                    run[query_id] = []
                run[query_id].append((doc_title, int(rank)))
                if i % 1000000 == 0:
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
                doc_id, doc_text = line.rstrip().split('\t')
                collection[doc_id] = doc_text.replace('\n', ' ')
                if i % 1000000 == 0:
                    print('Loading collection, doc {}'.format(i))
    return collection


def convert_train_dataset(train_dataset_path,
                        output_folder,
                        set_name,
                        do_udel=False,
                            ):

    print('Converting to Train to pairs tsv...')

    start_time = time.time()

    print('Counting number of examples...')
    num_lines = sum(1 for line in open(train_dataset_path, 'r'))
    print('{} examples found.'.format(num_lines))

    if do_udel:
        converter = UdelConverter()
    
    with open(f'{output_folder}/{set_name}_pairs.tsv', 'w') as out:
        with open(train_dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i % 1000 == 0:
                    time_passed = int(time.time() - start_time)
                    print('Processed training set, line {} of {} in {} sec'.format(
                        i, num_lines, time_passed))
                    hours_remaining = (num_lines - i) * time_passed / (max(1.0, i) * 3600)
                    print('Estimated hours remaining to write the training set: {}'.format(
                        hours_remaining))

                query, positive_doc, negative_doc = line.rstrip().split('\t')

                clean_query = clean_text(query)
                positive_doc = strip_html_xml_tags(positive_doc)
                positive_doc = clean_text(positive_doc)
                negative_doc = strip_html_xml_tags(negative_doc)
                negative_doc = clean_text(negative_doc)
                if do_udel:
                    clean_query = converter.convert_query_to_udel(clean_query)

                out.write('\t'.join([clean_query, positive_doc, '1'])+'\n')
                out.write('\t'.join([clean_query, negative_doc, '0'])+'\n')     

    print("writer closed, DONE !")
    print(f'writer closed with {i*2} lines')

class MsMarcoPassageProcessor(DataProcessor):

    def __init__(self, 
                passage_handle,
                marker
                ):
        super().__init__()
        self.passage_handle = passage_handle
        self.marker = marker

    def get_train_dataset (self, data_path, batch_size):
        return self.passage_handle.get_train_dataset(data_path, batch_size)
    
    def get_eval_dataset (self, data_path, batch_size, num_skip=0):
        return self.passage_handle.get_eval_dataset(data_path, batch_size, num_skip)

    def prepare_train_dataset( self,
                             data_path, 
                             output_dir,
                             set_name,
                             ):
        tf_writer = tf.io.TFRecordWriter(f"{output_dir}/dataset_{set_name}_train.tf")
        tsv_writer = open(f"{output_dir}/pairs_{set_name}_train.tsv", 'w')

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

                query, doc, label = line.rstrip().split('\t')
                q, p = self.marker.mark(query, doc)
                # write tfrecord
                self.passage_handle.write_train_example(tf_writer, q, [p], [int(label)])
                tsv_writer.write(f"{q}\t{p}\t{label}\n")
        tf_writer.close()
        tsv_writer.close()

    # def prepare_inference_dataset( self,
    #                          data_path, 
    #                          output_dir,
    #                          set_name, ):
    #     tf_writer = tf.io.TFRecordWriter(f"{output_dir}/dataset_{set_name}.tf")
    #     tsv_writer = open(f"{output_dir}/pairs_{set_name}.tsv", 'w')

    #     start_time = time.time()

    #     print('Counting number of examples...')
    #     num_lines = sum(1 for line in open(data_path, 'r'))
    #     print('{} examples found.'.format(num_lines))

    #     with open(data_path, 'r') as f:
    #         for i, line in enumerate(f):
    #             if i % 1000 == 0:
    #                 time_passed = int(time.time() - start_time)
    #                 print('Processed training set, line {} of {} in {} sec'.format(
    #                     i, num_lines, time_passed))
    #                 hours_remaining = (num_lines - i) * time_passed / (max(1.0, i) * 3600)
    #                 print('Estimated hours remaining to write the training set: {}'.format(
    #                     hours_remaining))
                
    #             qid, pid, query, doc, label, len_gt_query = line.rstrip().split('\t')
    #             q, p = self.marker.mark(query, doc)

    #             # write tfrecord
    #             self.passage_handle.write_eval_example(tf_writer, q, [p], [int(label)], qid, [pid], int(len_gt_query))
    #             tsv_writer.write(f"{qid}\t{q}\t{pid}\t{p}\t{label}\t{len_gt_query}\n")
    #     tf_writer.close()
    #     tsv_writer.close()

    def prepare_inference_dataset( self,
                             data_path, 
                             output_dir,
                             set_name,
                              ):
        tf_writer = tf.io.TFRecordWriter(f"{output_dir}/dataset_{set_name}.tf")
        tsv_writer = open(f"{output_dir}/pairs_{set_name}.tsv", 'w')
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
                              
                qid, pid, query, doc, label, len_gt_query = line.rstrip().split('\t')
                q, p = self.marker.mark(query, doc)

                # write tfrecord
                i_ids = self.passage_handle.write_eval_example(tf_writer, ids_writer, i_ids, q, [p], [int(label)], qid, [pid], int(len_gt_query))
                tsv_writer.write(f"{qid}\t{q}\t{pid}\t{p}\t{label}\t{len_gt_query}\n")
        tf_writer.close()
        tsv_writer.close()
        ids_writer.close()




