import collections
import os , time
import re, sys
from bs4 import BeautifulSoup
import functools


from .convert_to_udel import UdelConverter


def clean_output_text(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)
        if not isinstance(results, tuple):
            results = (results,)
        clean_results = ()
        for item in results: 
            clean_results += (clean_text(strip_html_xml_tags(item)),)
        return (clean_results if len(clean_results)>1 else clean_results[0])
    return wrapper

""" Cleaning utilities """

def clean_text(text):
    #encoding
    try:
        t = text.encode("ISO 8859-1")
        enc_text = t.decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError) as e:
        enc_text = text
    
    #
    text = re.sub("’","'",text)
    #empty characters
    text = " ".join(text.strip().split())

    return text

def strip_html_xml_tags(text):
    return BeautifulSoup(text, "lxml").text

""" Intermediate corpus format converter """
class DataPrep(object):

    def convert_eval_dataset(
        self,
        args,
    ):
        """ creates a .tsv file in a unified format """
        raise NotImplementedError ()

    def convert_train_dataset(
        self,
        args,
    ):
        """ creates a .tsv file in a unified format """
        raise NotImplementedError ()

class TopKPrepFromRetriever(DataPrep):

    def _load_queries(self, path):
        """Loads queries into a dict of key: query_id, value: query text."""
        queries = {}
        with open(path) as f:
            for i, line in enumerate(f):
                    query_id, query = line.rstrip().split('\t')
                    queries[query_id] = query
                    if i % 10 == 0:
                        print(f'Loading queries {i}')
        return queries
    
    def _load_collection(self, path):
        """Loads tsv collection into a dict of key: doc id, value: (title, body)."""
        collection = {}
        with open(path) as f:
            for i, line in enumerate(f):
                    doc_id, doc_title, doc_body = line.rstrip('\n').split('\t')

                    body = strip_html_xml_tags(doc_body)
                    clean_body = clean_text(body)
                    title = strip_html_xml_tags(doc_title)
                    clean_title = clean_text(title)

                    collection[doc_id] = (clean_title, clean_body)
                    if i % 10000 == 0:
                        print(f'Loading collection, doc {i}')
        return collection
    
    def _load_run(self, path):
        raise NotImplementedError ()

    def _merge(self, qrels, run, queries):
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
    
    def _convert_dataset(
            self,
            data, 
            collection, 
            set_name, 
            num_eval_docs, 
            output_dir
    ):
        raise NotImplementedError ()
    
class TRECDocumentPrepFromRetriever(TopKPrepFromRetriever):

    def __init__(
        self, 
        how ='tokens', 
        plen=150, 
        overlap=50, 
        tlen=-1,
        max_pass_per_doc = 30,
    ):
        super().__init__()
        self.split = (how=='words')
        self.plen = plen
        self.overlap = overlap
        self.tlen = tlen
        self.max_pass_per_doc = max_pass_per_doc

        self.stats = dict()

    def convert_eval_dataset(
        self,
        args,
    ):
        print('Begin...')
        if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)

        queries = self._load_queries(path=args.queries_path)
        run, qrels = self._load_run(path=args.run_path)
        data = self._merge(qrels=qrels, run=run, queries=queries)

        print('Loading Collection...')
        collection = self._load_collection(args.collection_path)

        if self.split:
            collection = self._split_docs(collection)
        print('Saving Data File...')
        self._convert_dataset(data, collection, args.set_name, args.num_eval_docs, args.output_dir)

        print('Done!')
        return self.stats

    def _convert_dataset(
            self,
            data, 
            collection, 
            set_name, 
            num_eval_docs, 
            output_dir
    ):
        suff = 'passages' if self.split else 'doc'
        output_path = os.path.join(output_dir, f'run_{set_name}_{suff}.tsv')
        start_time = time.time()
        random_title = list(collection.keys())[0]
        
        with open(output_path, 'w') as doc_writer:
            for idx, query_id in enumerate(data):
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
                        if self.split :
                            passages = collection[doc_title]
                            self.stats[doc_title] = len(passages)
                            for pos, p in passages.items():
                                id_pass = f'{doc_title}_passage-{pos}'
                                doc_writer.write("\t".join((query_id, id_pass, clean_query,
                                                    p, str(label), str(len_gt_query))) + "\n")
                        else: 
                            title, doc = collection[doc_title]
                            doc_writer.write("\t".join((query_id, doc_title, clean_query, title,
                                                    clean_doc, str(label), str(len_gt_query))) + "\n")

                    if idx % 10 == 0:
                        print(f'Wrote {idx} of {len(data)} queries')
                        time_passed = time.time() - start_time
                        est_hours = (len(data) - idx) * time_passed / (max(1.0, idx) * 3600)
                        print(f'Estimated total hours to save: {est_hours}')


    def _split_docs(self, collection):
        pass_collection = dict()
        for did in collection:
            title, doc = collection[did]
            passages = self._split_doc(title, doc)
            pass_collection[did] = passages
        return pass_collection

    def _split_doc(self, title, doc):
        """ Modified from https://github.com/canjiali/PARADE/blob/master/generate_data.py
        :param title: str
        :param doc: str
        :return: Dict[]
        """
        title_words = title.strip().split(' ')
        doc_words = doc.strip().split(' ')

        trunc_title = title_words[:self.tlen]
        if len(trunc_title)>0 and trunc_title[-1] != '.':
            trunc_title.append('.')

        pos, idx_start, idx_end = 0, 0, 0
        passages = dict()
        real_plen = self.plen - len(trunc_title)
        trunc_title = ' '.join(trunc_title)
        while idx_start < len(doc_words):
            idx_end = idx_start + real_plen
            if idx_end >= len(doc_words):
                idx_end = len(doc_words)
            # if the last one is shorter than 'overlap', it is already in the previous passage.
            if len(passages) > 0 and idx_end - idx_start <= self.overlap:
                break
            p = trunc_title + ' ' + ' '.join(doc_words[idx_start:idx_end])
            passages[pos] = p
            pos += 1
            idx_start = idx_start + real_plen - self.overlap

        if len(passages) > self.max_pass_per_doc:
            chosen_ids = sorted(random.sample(range(1, len(passages) - 1), self.max_pass_per_doc - 2))
            chosen_ids = [0] + chosen_ids + [len(passages) - 1]
            passages = {idx:passages[idx] for idx in chosen_ids}
        return passages

    def _load_run(self, path):
        """Loads run into a dict of key: query_id, value: list of candidate doc ids."""
        # We want to preserve the order of runs so we can pair the run file with the
        # TFRecord file.
        run = collections.OrderedDict()
        qrels = collections.defaultdict(set)

        relevance_threshold = 1

        with open(path) as f:
            for i, line in enumerate(f):
                    query_id, doc_title, pred, rank, relevance = line.split('\t')
                    if query_id not in run:
                        run[query_id] = []
                    run[query_id].append((doc_title, int(rank)))
                    if int(relevance) >= relevance_threshold:
                        qrels[query_id].add(doc_title)
                    if i % 1000 == 0:
                        print('Loading run {}'.format(i))
        # Sort candidate docs by rank.
        sorted_run = collections.OrderedDict()
        for query_id, doc_titles_ranks in run.items():
                sorted(doc_titles_ranks, key=lambda x: x[1])
                doc_titles = [doc_titles for doc_titles, _ in doc_titles_ranks]
                sorted_run[query_id] = doc_titles

        return sorted_run, qrels


class MsMarcoPassagePrep(TopKPrepFromRetriever):

    def convert_eval_dataset(
        self,
        args,
    ):
        if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)

        qrels = self._load_qrels(args.qrels_path)

        queries = self._load_queries(path=args.queries_path)
        run = self._load_run(path=args.run_path)
        data = self._merge(qrels=qrels, run=run, queries=queries)

        print('Loading Collection...')
        collection = self._load_collection(args.collection_path)

        print('Converting to TFRecord...')
        self._convert_dataset(data,collection, args.set_name,
                            args.num_eval_docs, args.output_dir)

        print('Done!')

    def _convert_dataset(
            self,
            data, 
            collection, 
            set_name, 
            num_eval_docs, 
            output_dir
    ):

        output_path = os.path.join(output_dir, f'run_{set_name}_full.tsv')
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
                        print(f'wrote {i} of {len(data)} queries')
                        time_passed = time.time() - start_time
                        est_hours = (len(data) - i) * time_passed / (max(1.0, i) * 3600)
                        print(f'estimated total hours to save: {est_hours}')


    def _load_qrels(self, path, train_set=False):
        """Loads qrels into a dict of key: query_id, value: list of relevant doc ids."""
        qrels = collections.defaultdict(set)

        relevance_threshold = 1 if train_set else 2

        with open(path) as f:
            for i, line in enumerate(f):
                    query_id, _, doc_id, relevance = line.rstrip().split('\t')
                    if int(relevance) >= relevance_threshold:
                        qrels[query_id].add(doc_id)
                    if i % 1000 == 0:
                        print('Loading qrels {}'.format(i))
        return qrels

    def _load_run(self, path):
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

    def convert_train_dataset(
            self,
            args,
    ):

        print('Converting Train to pairs tsv...')

        start_time = time.time()

        print('Counting number of examples...')
        num_lines = sum(1 for line in open(args.dataset_path, 'r'))
        print(f'{num_lines} examples found.')

        ''' this argument is not in prep_data'''
        if args.do_udel: 
            converter = UdelConverter()
        
        with open(os.path.join(args.output_dir, f'{args.set_name}_pairs.tsv'), 'w') as out:
            with open(args.dataset_path, 'r') as f:
                for i, line in enumerate(f):
                    if i % 1000 == 0:
                        time_passed = int(time.time() - start_time)
                        print(f'Processed training set, line {i} of {num_lines} in {time_passed} sec')
                        hours_remaining = (num_lines - i) * time_passed / (max(1.0, i) * 3600)
                        print(f'Estimated hours remaining to write the training set: {hours_remaining}')

                    query, positive_doc, negative_doc = line.rstrip().split('\t')

                    clean_query = clean_text(query)
                    positive_doc = strip_html_xml_tags(positive_doc)
                    positive_doc = clean_text(positive_doc)
                    negative_doc = strip_html_xml_tags(negative_doc)
                    negative_doc = clean_text(negative_doc)

                    if args.do_udel:
                        clean_query = converter.convert_query_to_udel(clean_query)

                    out.write('\t'.join([clean_query, positive_doc, '1'])+'\n')
                    out.write('\t'.join([clean_query, negative_doc, '0'])+'\n')     

        print('Writer closed, DONE !')
        print(f'Writer closed with {(i+1)*2} lines.')
