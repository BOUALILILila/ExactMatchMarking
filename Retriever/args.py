from argparse import ArgumentParser

TOPIC_FIELDS=['title','description', 'desc_key_words', 'title_desc']

def get_args():
    parser = ArgumentParser(description='Retriever')


    parser.add_argument('--data_path', default='data')
    parser.add_argument('--anserini_path', help='Path to Anserini root')
    parser.add_argument('--collection', default='robust04', help=f'{TOPIC_FIELDS}')

    parser.add_argument('--topic_field', default='title', help='[title, description, desc_key_words, title_desc]')
    parser.add_argument('--use_doc_title', default=False, help='Whether include the document title or not')
    parser.add_argument('--K', type=int, default=1000, help='Top-k documents to retrieve')
    parser.add_argument('--rm3', default=True, help='Use RM3')

    parser.add_argument('--index_path', default='lucene-index.robust04.pos+docvectors+rawdocs', help='Path to Lucene index')
    args, other = parser.parse_known_args()
    return args, other
