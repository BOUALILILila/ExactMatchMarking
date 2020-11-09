"""
    create an MBPrep calss in data_utils and copy the two convert functions and make sure its arguments are in args in prep_data.py script
"""
import logging
import collections
import os, time

from .processor_utils import DataProcessor
from .data_utils import strip_html_xml_tags, clean_text
from . import MsMarcoPassageProcessor 


logger = logging.getLogger(__name__)


def convert_eval_dataset(dataset_path,
                        output_folder,
                        set_name,):

    print('Converting to Eval to pairs tsv...')

    start_time = time.time()

    print('Counting number of examples...')
    num_lines = sum(1 for line in open(f"{dataset_path}/sim.txt", 'r'))
    print('{} examples found.'.format(num_lines))
    
    fa = open(os.path.join(dataset_path,'a.toks')) # queries
    fb = open(os.path.join(dataset_path,'b.toks')) # tweets
    fsim = open(os.path.join(dataset_path,'sim.txt')) # labels
    fid = open(os.path.join(dataset_path,'id.txt')) # ids

    with open(f'{output_folder}/mb_{set_name}_pairs.tsv', 'w') as out:
        for i, (a, b , label, ids) in enumerate(zip(fa, fb, fsim, fid)):
            if i % 1000 == 0:
                time_passed = int(time.time() - start_time)
                print('Processed training set, line {} of {} in {} sec'.format(
                        i, num_lines, time_passed))
                hours_remaining = (num_lines - i) * time_passed / (max(1.0, i) * 3600)
                print('Estimated hours remaining to write the training set: {}'.format(
                        hours_remaining))

            clean_query = clean_text(a)
            clean_tweet = strip_html_xml_tags(b)
            clean_tweet = clean_text(clean_tweet)

            label = label.rstrip()
            qid, _, tweetid, _, _, _ = ids.rstrip().split()

            out.write('\t'.join([qid, tweetid, clean_query, clean_tweet, label, '0'])+'\n')   # label has \n  

    print("writer closed, DONE !")
    print(f'writer closed with {i} lines')


def convert_train_dataset(train_dataset_path,
                        output_folder,
                        set_name,
                            ):

    print('Converting to Train to pairs tsv...')

    start_time = time.time()

    print('Counting number of examples...')
    num_lines = sum(1 for line in open(f"{train_dataset_path}/sim.txt", 'r'))
    print('{} examples found.'.format(num_lines))
    
    fa = open(os.path.join(train_dataset_path,'a.toks')) # queries
    fb = open(os.path.join(train_dataset_path,'b.toks')) # tweets
    fsim = open(os.path.join(train_dataset_path,'sim.txt')) # labels

    with open(f'{output_folder}/mb_{set_name}_pairs.tsv', 'w') as out:
        for i, (a, b , label) in enumerate(zip(fa, fb, fsim)):
            if i % 1000 == 0:
                time_passed = int(time.time() - start_time)
                print('Processed training set, line {} of {} in {} sec'.format(
                        i, num_lines, time_passed))
                hours_remaining = (num_lines - i) * time_passed / (max(1.0, i) * 3600)
                print('Estimated hours remaining to write the training set: {}'.format(
                        hours_remaining))

            clean_query = clean_text(a)
            clean_tweet = strip_html_xml_tags(b)
            clean_tweet = clean_text(clean_tweet)

            out.write('\t'.join([clean_query, clean_tweet, label]))   # label has \n  

    print("writer closed, DONE !")
    print(f'writer closed with {i} lines')

class MbProcessor(MsMarcoPassageProcessor):

    def __init__(self, 
                passage_handle,
                marker
                ):
        super(MbProcessor,self).__init__(passage_handle, marker)
    
