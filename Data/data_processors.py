import os, time
import tensorflow as tf
from .io_handles import DocumentHandle, PassageHandle
from .marker_utils import Marker

class DataProcessor(object):  
    
    def get_train_dataset (self, data_path, batch_size):
        """ Reads a TFRecord dataset """
        raise NotImplementedError()

    def get_eval_dataset (self, data_path, batch_size, num_skip=0):
        """ Reads a TFRecord file containing eval examples and returns a TF dataset"""
        raise NotImplementedError ()
    
    def prepare_train_dataset(
                        self,
                        data_path, 
                        output_dir,
    ):
        """ preprocess the raw training data and save it as pairs/tfrecord """
        raise NotImplementedError()
    
    def prepare_inference_dataset(
                            self,
                            data_path, 
                            output_dir,
                            set_name,
    ):
        """ preprocess the raw inference data from a run file and save it as pairs/tfrecord """
        raise NotImplementedError()

class DocumentProcessor(DataProcessor):

    def __init__(self, 
                document_handle: DocumentHandle,
                marker: Marker,
                ):
        super().__init__()
        self.handle = document_handle
        self.marker = marker
    
    def get_eval_dataset (self, data_path, batch_size, num_skip=0):
        return self.handle.get_eval_dataset(data_path, batch_size, num_skip)

    def prepare_inference_dataset(
                        self,
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

class PassageProcessor(DataProcessor):

    def __init__(self, 
                passage_handle: PassageHandle,
                marker: Marker,
                ):
        super().__init__()
        self.passage_handle = passage_handle
        self.marker = marker

    def get_train_dataset (self, data_path, batch_size):
        return self.passage_handle.get_train_dataset(data_path, batch_size)
    
    def get_eval_dataset (self, data_path, batch_size, num_skip=0):
        return self.passage_handle.get_eval_dataset(data_path, batch_size, num_skip)

    def prepare_train_dataset(
                        self,
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

    def prepare_inference_dataset(
                        self,
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

