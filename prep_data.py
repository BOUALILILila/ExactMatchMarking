import argparse
import os
from Data import (
    TRECDocumentPrepFromRetriever, 
    MsMarcoPassagePrep, 
    get_collection_prep, 
    get_collection_names,
)


def main():
    
    parser = argparse.ArgumentParser(description='PairsDataPrep')

    parser.add_argument('--collection', type=str, required=True, help=f'{get_collection_names}')
    parser.add_argument('--set', type=str, default='test')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--queries_path', type=str, required=False,
                            help='The path to the test queries .tsv file: q_id, query')
    parser.add_argument('--run_path', type=str, required=False,
                            help='The path to the run file .tsv file : q_id, doc_id, score, rank, judgement.')
    parser.add_argument('--collection_path', type=str, required=False,
                            help='The path to the documents .tsv file: doc_id, title, body.')
    parser.add_argument('--set_name', type=str, required=True,
                            help='Name of the experiment.')
    parser.add_argument('--num_eval_docs', default=1000, type=int,
                            help='The number of documents retrieved per query.')
    
    parser.add_argument('--dataset_path', type=str, required=False)
    parser.add_argument('--do_udel', type=bool, default=False)

    args, other = parser.parse_known_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        

    prep = get_collection_prep(args.collection)()
    

    if args.set == 'test':
        prep.convert_eval_dataset(args)
    
    elif args.set == 'train':
        prep.convert_train_dataset(args)

if __name__ == '__main__':
    main()
