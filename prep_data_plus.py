import argparse
import os
import json

from Data import get_available_collections, get_collection, DocumentCol


def main():
    
    parser = argparse.ArgumentParser(description='PairsDataPrep')

    parser.add_argument('--collection', type=str, required=True, help=f'{get_available_collections()}')
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
    parser.add_argument('--do_udel', action='store_true', default=False)

    parser.add_argument('--how', type=str, default='tokens', required=False)
    parser.add_argument('--plen', default=300, type=int,
                            help='If the passages are created over words split, this indicates how many words per passage.')
    parser.add_argument('--overlap', default=100, type=int,
                            help='If the passages are created over words split, this indicates how many overlap words between consecutive passages.')
    parser.add_argument('--tlen', default=0, type=int,
                            help='If the passages are created over words split, this indicates how many words per title.')
    parser.add_argument('--max_pass_per_doc', default=10000, type=int,
                            help='If the passages are created over words split, this indicates how many passages per document.')


    args, other = parser.parse_known_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    col = get_collection(args.collection, args.how)

    if isinstance(col, DocumentCol):
        prep = col.get_prep(args.plen, args.overlap, args.tlen, args.max_pass_per_doc)
    else:
        prep = col.get_prep()
    
    if args.set in ('test', 'dev'):
        stats = prep.convert_eval_dataset(args)
    elif args.set == 'train':
        prep.convert_train_dataset(args)
    else :
        raise ValueError("Set must be in ['train', 'dev', 'test] !")


    print(stats)
    with open(os.path.join(args.output_dir,f'prep_stats_{args.set_name}.json'), 'w') as fp:
        json.dump(stats, fp)


if __name__ == '__main__':
    main()
