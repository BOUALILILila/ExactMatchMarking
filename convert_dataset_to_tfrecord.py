import argparse
import os
from transformers import AutoTokenizer, BertTokenizerFast

from Data import get_available_collections, get_collection, get_marker


def main():
    
    parser = argparse.ArgumentParser(description='SaveDatasetTFRecord')

    ## Required parameters
    parser.add_argument('--collection', type=str, required=True, help=f'{get_available_collections()}')
    parser.add_argument('--set', type=str, default='test')
    parser.add_argument("--strategy", default=None, type=str, required=True,
                            help="the marking strategy in ('base', 'sim_doc', 'sim_pair', 'pre_doc', 'pre_pair')")
    parser.add_argument("--data_path", default=None, type=str, required=True,
                            help="The input data file.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                            help="The output dir, data to be saved.")
    parser.add_argument("--tokenizer_name_path", default='bert-base-uncased', type=str, required=False,
                            help=f"Path to the initialization tokenizer or name in transformers")
    parser.add_argument("--max_seq_len", default=512, type=int, required=False,
                            help=" Max seq length for the transformer.")
    parser.add_argument("--max_query_len", default=64, type=int, required=False,
                            help=" Max query length in the input sequence of the transformer.")
    parser.add_argument("--seed", default=42, type=int, required=False,
                            help=" Random seed.")
    parser.add_argument("--max_title_len", default=64, type=int, required=False,
                            help=" Max title length in the input sequence of the transformer.")
    parser.add_argument("--chunk_size", default=384, type=int, required=False,
                            help=" Split the sequence into fixed size chunks of size chunk_size.")
    parser.add_argument("--stride", default=192, type=int, required=False,
                            help="If split into overlapping chunks set this stride.")
    parser.add_argument("--set_name", default=None, type=str, required=True,
                            help="Set name.")

    args, other = parser.parse_known_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    col = get_collection(args.collection, how='words')

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name_path)
    
    marker = get_marker(args.strategy.lower())

    processor = col.get_processor(args.max_seq_len, args.max_query_len,
                            max_title_length = args.max_title_len, 
                            chunk_size = args.chunk_size, 
                            stride = args.stride)


    if args.set in ('test', 'dev'):
        processor.prepare_inference_dataset(tokenizer, marker, args.data_path, args.output_dir, f'{args.set}_{args.set_name}')
    elif args.set == 'train':
        processor.prepare_train_dataset(tokenizer, marker, args.data_path, args.output_dir, f'{args.set}_{args.set_name}')
    else :
        raise ValueError("Set must be in ['train', 'dev', 'test] !")

if __name__ == "__main__":
    main()
