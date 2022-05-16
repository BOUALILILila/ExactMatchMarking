import os, time
import typing
import tensorflow as tf
from transformers import PreTrainedTokenizer, AutoTokenizer

from .io_handles import DocumentHandle, PassageHandle, TFRecordHandle
from .marker_utils import Marker

class DataProcessor(object): 

    def __init__(
        self, 
        handle: TFRecordHandle,
    ):
        self.handle = handle
     
    
    def get_train_dataset (self, data_path: str, batch_size: int, seed: int = 42):
        """ Reads a TFRecord dataset """
        raise NotImplementedError()

    def get_eval_dataset (self, data_path: str, batch_size: int, num_skip: int = 0):
        """ Reads a TFRecord file containing eval examples and returns a TF dataset"""
        raise NotImplementedError ()
    
    def prepare_train_dataset(
                        self,
                        tokenizer: typing.Union[PreTrainedTokenizer, AutoTokenizer],
                        marker: Marker,
                        data_path: str, 
                        output_dir: str,
                        set_name: str,
    ):
        """ preprocess the raw training data and save it as pairs/tfrecord """
        raise NotImplementedError()
    
    def prepare_inference_dataset(
                            self,
                            tokenizer: typing.Union[PreTrainedTokenizer, AutoTokenizer],
                            marker: Marker,
                            data_path: str, 
                            output_dir: str,
                            set_name: str,
    ):
        """ preprocess the raw inference data from a run file and save it as pairs/tfrecord """
        raise NotImplementedError()

class DocumentProcessor(DataProcessor):

    def __init__(self, 
                handle: DocumentHandle,
                ):
        super().__init__(handle)
    
    def get_eval_dataset(self, data_path: str, batch_size: int, num_skip: int = 0):
        return self.handle.get_eval_dataset(data_path, batch_size, num_skip)

    def prepare_inference_dataset(
                        self,
                        tokenizer: typing.Union[PreTrainedTokenizer, AutoTokenizer],
                        marker: Marker,
                        data_path: str, 
                        output_dir: str,
                        set_name: str,
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
                              
                qid, did, query, title ,doc, label, len_gt_query = line.rstrip().split('\t')

                m_query, m_title, m_doc = marker.mark(query, title, doc)

                # write tfrecord
                i_ids = self.handle.write_eval_example(tf_writer, tokenizer, 
                                m_query, [(m_title,m_doc)], [int(label)], 
                                ids_writer, i_ids, qid, [did], int(len_gt_query)) 
                tsv_writer.write(f"{qid}\t{m_query}\t{did}\t{m_title}\t{m_doc}\t{label}\t{len_gt_query}\n")
        tf_writer.close()
        tsv_writer.close()
        ids_writer.close()

class PassageProcessor(DataProcessor):

    def __init__(self, 
                handle: PassageHandle,
                ):
        super().__init__(handle)

    def get_train_dataset (self, data_path: str, batch_size: int, seed: int = 42):
        return self.handle.get_train_dataset(data_path, batch_size, seed)
    
    def get_eval_dataset (self, data_path: str, batch_size: int, num_skip: int = 0):
        return self.handle.get_eval_dataset(data_path, batch_size, num_skip)

    def prepare_train_dataset(
                        self,
                        tokenizer: typing.Union[PreTrainedTokenizer, AutoTokenizer],
                        marker: Marker,
                        data_path: str, 
                        output_dir: str,
                        set_name: str,
    ):
        tf_writer = tf.io.TFRecordWriter(f"{output_dir}/dataset_{set_name}.tf")
        #tsv_writer = open(f"{output_dir}/pairs_{set_name}.tsv", 'w')

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
                q, p = marker.mark(query, doc)
                # write tfrecord
                self.handle.write_train_example(tf_writer, tokenizer, q, [p], [int(label)])
                #tsv_writer.write(f"{q}\t{p}\t{label}\n")
        tf_writer.close()
        #tsv_writer.close()

    def prepare_inference_dataset(
                        self,
                        tokenizer: typing.Union[PreTrainedTokenizer, AutoTokenizer],
                        marker: Marker,
                        data_path: str, 
                        output_dir: str,
                        set_name: str,
    ):
        tf_writer = tf.io.TFRecordWriter(f"{output_dir}/dataset_{set_name}.tf")
        tsv_writer = open(f"{output_dir}/pairs_{set_name}.tsv", 'w')
        ids_writer = open(f"{output_dir}/query_pass_ids_{set_name}.tsv", 'w')
        i_ids = 0

        start_time = time.time()

        print('Counting number of examples...')
        num_lines = sum(1 for line in open(data_path, 'r'))
        print('{} examples found.'.format(num_lines))

        #
        prev_did = None
        all_pass = ''
        pids = []
        passages =[]
        labels =[]
        #
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

                q, p = marker.mark(query, doc)

                # write tfrecord
                i_ids = self.handle.write_eval_example(tf_writer, tokenizer,
                             q, [p], [int(label)], ids_writer, i_ids, 
                             qid, [pid], int(len_gt_query))
                
                tsv_writer.write(f"{qid}\t{q}\t{pid}\t{p}\t{label}\t{len_gt_query}\n")
        tf_writer.close()
        tsv_writer.close()
        ids_writer.close()

    def prepare_inference_dataset_doc_level(
                        self,
                        tokenizer: typing.Union[PreTrainedTokenizer, AutoTokenizer],
                        marker: Marker,
                        data_path: str, 
                        output_dir: str,
                        set_name: str,
    ):
        tf_writer = tf.io.TFRecordWriter(f"{output_dir}/dataset_{set_name}_doc_level_marking.tf")
        #tsv_writer = open(f"{output_dir}/pairs_{set_name}_doc_level_marking.tsv", 'w')
        ids_writer = open(f"{output_dir}/query_pass_ids_{set_name}_doc_level_marking.tsv", 'w')
        i_ids = 0

        start_time = time.time()

        print('Counting number of examples...')
        num_lines = sum(1 for line in open(data_path, 'r'))
        print('{} examples found.'.format(num_lines))

        #
        SEP ='[my_passage_separator]'

        with open(data_path, 'r') as f:
            line = f.readline()
            qid, pid, query, doc, label, len_gt_query = line.rstrip().split('\t')
            prev_record = [qid, pid.split('_passage-')[0], query, len_gt_query]
            all_pass = doc + ' ' + SEP +' '
            pids = [pid]
            labels = [int(label)]

            for i, line in enumerate(f):
                if i % 1000 == 0:
                    time_passed = int(time.time() - start_time)
                    print('Processed training set, line {} of {} in {} sec'.format(
                                i, num_lines, time_passed))
                    hours_remaining = (num_lines - i) * time_passed / (max(1.0, i) * 3600)
                    print('Estimated hours remaining to write the training set: {}'.format(
                                hours_remaining))

                qid, pid, query, doc, label, len_gt_query = line.rstrip().split('\t')
                        
                #
                did, pass_pos = pid.split('_passage-')
                prev_did = prev_record[1]

                #if did != prev_did or qid != prev_record[0]:
                if pass_pos == '0': # tests are in test_code dir .ipynb
                    q, all_pass = marker.mark(prev_record[2], all_pass)
                    passages = all_pass.split(SEP)
                    i_ids = self.handle.write_eval_example(tf_writer, tokenizer,
                                    q, passages, labels, ids_writer, i_ids, 
                                    prev_record[0], pids, int(prev_record[-1]))
                    # for i, pass_id in enumerate(pids):
                    #     tsv_writer.write(f"{prev_record[0]}\t{q}\t{pass_id}\t{passages[i]}\t{labels[i]}\t{prev_record[-1]}\n")
                    # next
                    all_pass =''
                    pids = []
                    labels =[]
                    prev_record = [qid, did, query, len_gt_query]
                        
                all_pass += doc + ' ' + SEP +' '
                pids.append(pid)
                labels.append(int(label))
        # last
        q, all_pass = marker.mark(query, all_pass)
        passages = all_pass.split(SEP)
        i_ids = self.handle.write_eval_example(tf_writer, tokenizer,
                                    q, passages, labels, ids_writer, i_ids, 
                                    qid, pids, int(len_gt_query))
        # for i, pass_id in enumerate(pids):
        #     tsv_writer.write(f"{qid}\t{q}\t{pass_id}\t{passages[i]}\t{labels[i]}\t{len_gt_query}\n")
                        
        tf_writer.close()
        #tsv_writer.close()
        ids_writer.close()


