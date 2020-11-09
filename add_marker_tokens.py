import torch
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def main():
    
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--save_dir", default=None, type=str, required=True,
                            help="The output dir, data to be saved.")
    parser.add_argument("--tokenizer_name_path", default='bert-base-uncased', type=str, required=False,
                            help=f"path to the initialization tokenizer or name in transformers")
    parser.add_argument("--name", default=None, type=str, required=True,
                            help="The name for the new extended vocabulary and model.")

    args = parser.parse_args()

    init_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_path)

    nb_tokens = 20 # only 6 queries out of all the train queries have more than 20 stems 

    model = AutoModelForSequenceClassification.from_pretrained(args.tokenizer_name_path)

    # add the tokens to the tokenizer --vocab
    embed_1 = model.bert.embeddings.word_embeddings.weight[1031, :] # [
    embed_2 = model.bert.embeddings.word_embeddings.weight[1041, :] # e
    embed_3 = model.bert.embeddings.word_embeddings.weight[1033, :] # ]
    embed_4 = model.bert.embeddings.word_embeddings.weight[1032, :] # \
    l1 = [embed_1, embed_2, embed_3]
    l2 = [embed_1, embed_4, embed_2, embed_3]
    for i in range(nb_tokens):
        token_id = init_tokenizer.encode_plus(f'e{i}', add_special_tokens= False)['input_ids'][1]
        embed = model.bert.embeddings.word_embeddings.weight[token_id, :]
        
        tokenizer.add_tokens(f'[e{i}]')
        model.resize_token_embeddings(len(tokenizer))
        model.bert.embeddings.word_embeddings.weight[-1, :] = torch.mean(torch.stack([embed_1, embed_2, embed, embed_3]), axis = 0)

        tokenizer.add_tokens(f'[\e{i}]')
        model.resize_token_embeddings(len(tokenizer))
        model.bert.embeddings.word_embeddings.weight[-1, :] = torch.mean(torch.stack([embed_1, embed_4, embed_2, embed, embed_3]), axis = 0)

    # Save
    print(f'saving to {args.save_dir}/pre_model_{args.name}, pre_tokenizer_{args.name}')
    model.save_pretrained(f"{args.save_dir}/pre_model_{args.name}")  

    tokenizer.save_pretrained(f"{args.save_dir}/pre_tokenizer_{args.name}") 

    print('Done !')

if __name__ == "__main__":
    main()
