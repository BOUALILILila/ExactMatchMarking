import os
from argparse import ArgumentParser

from Retriever import retrieve
from Data import get_available_collections, get_collection

os.environ['PATH']=os.environ['PATH']+':/logiciels/jdk-13.0.1/bin'

#python "data/data/Repositories/MarkedBERT/first_stage/retrieve.py" 
# --collection core18  --data_path data/data/datasets 
# --topic_field description_key_words  
# --anserini_path anserini --index anserini/indexes/lucene-index.core18.pos+docvectors+raw

TOPIC_FIELDS=['title','description']

def main():
    
    parser = ArgumentParser(description='Retriever')


    parser.add_argument('--data_path', default='data')
    parser.add_argument('--anserini_path', help='Path to Anserini root')
    parser.add_argument('--collection', default='robust04', help=f'{get_available_collections()}')

    parser.add_argument('--topic_field', default='title', help=TOPIC_FIELDS)
    parser.add_argument('--use_doc_title', action='store_true', default=False, help='Whether include the document title or not')
    parser.add_argument('--K', type=int, default=1000, help='Top-k documents to retrieve')
    parser.add_argument('--rm3', action='store_true', default=False, help='Use RM3')

    parser.add_argument('--index_path', default='lucene-index.robust04.pos+docvectors+rawdocs', help='Path to Lucene index')
    args, other = parser.parse_known_args()

    if args.topic_field.lower() not in TOPIC_FIELDS:
        raise ValueError(f'Unknown topic field: {args.topic_field}. Topic field must be in:{TOPIC_FIELDS}')

    col = get_collection(args.collection)

    retrieve(col, args)

if __name__ == '__main__':
    main()