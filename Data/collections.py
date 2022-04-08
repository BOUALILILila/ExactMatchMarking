import json, sys
from collections import defaultdict
from typing import Optional, Tuple
from bs4 import BeautifulSoup, re

from .data_utils import (
    MsMarcoPassagePrep, 
    TRECDocumentPrepFromRetriever, 
    MsMarcoDocumentPrep,
    DataPrep,
    clean_output_text)
from .io_handles import PassageHandle, DocumentSplitterHandle
from .data_processors import PassageProcessor, DocumentProcessor, DataProcessor

def get_available_collections():
    return list(COLLECTIONS.keys())

def get_collection(collection_name, how='words'):
    collection = collection_name.lower()
    if collection in COLLECTIONS.keys():
        return COLLECTIONS[collection](how)
    else :
        raise ValueError(f'Unrecognized collection {collection_name}')


class Collection(object):

    def __init__(self, name):
        self.col_name = name

    def name(self) -> str:
        return self.col_name
    
    def parse_queries(self, queries_path):
        raise NotImplementedError()

    def parse_doc(self, content):
        raise NotImplementedError()

    def get_prep(self) -> DataPrep:
        raise NotImplementedError()

    def get_processor(self, max_seq_len = 512, max_query_len = 64, **kwargs) -> DataProcessor:
        raise NotImplementedError()

class MsMarcoPassage(Collection):

    def __init__(self, *args):
        super().__init__('MsMarco')

    def parse_queries(self, queries_path):
        pass
    
    @clean_output_text
    def parse_doc(self, content):
        return content

    def get_prep(self):
        return MsMarcoPassagePrep()
    
    def get_processor(self, max_seq_len = 512, max_query_len = 64, **kwargs):
        handle = PassageHandle(max_seq_len, max_query_len, **kwargs)
        return PassageProcessor(handle)

class DocumentCol(Collection):

    @clean_output_text
    def parse_doc(self, content, use_doc_title = False)-> Tuple[str,str]:
        raise NotImplementedError()
    
    def get_processor(self, max_seq_len = 512, max_query_len = 64, **kwargs):
        if self.how == 'words':
            handle = PassageHandle(max_seq_len, max_query_len, **kwargs)
            return PassageProcessor(handle)
        else:
            handle = DocumentSplitterHandle(max_seq_len, max_query_len, **kwargs)
            return DocumentProcessor(handle)


class TRECCollection(DocumentCol):

    def parse_queries(self, queries_path):
        """  
        parse trec topics from https://github.com/capreolus-ir/capreolus/blob/413b3bacc5cb9afd6c36e465a804d30456cb31a4/capreolus/utils/trec.py
        :param queries_path:
        :return:
        """
        title, desc, narr = defaultdict(list), defaultdict(list), defaultdict(list)
        block = None
        if queries_path.endswith(".gz"):
            openf = gzip.open
        else:
            openf = open
        with openf(queries_path, "rt") as f:
            for line in f:
                line = line.strip()
                if line.startswith("<num>"):
                    # <num> Number: 700
                    qid = line.split()[-1]
                    if qid == '</num>':
                        qid = line.split()[-2]
                    qid = str(qid)
                    # no longer an int
                    # assert qid > 0
                    block = None
                elif line.startswith("<title>"):
                    # <title>  query here
                    title[qid].extend(line.strip().split()[1:])
                    if len(title[qid])>0:
                        if title[qid][0] == 'Topic:':
                            title[qid].pop(0)
                    block = "title"
                elif line.startswith("<desc>"):
                    # <desc> description \n description
                    desc[qid].extend(line.strip().split()[1:])
                    if len(desc[qid])>0:
                        if desc[qid][0] == 'Description:':
                            desc[qid].pop(0)
                    block = "desc"
                elif line.startswith("<narr>"):
                    # same format as <desc>
                    narr[qid].extend(line.strip().split()[1:])
                    if len(narr[qid])>0:
                        if narr[qid][0] == 'Narrative:':
                            narr[qid].pop(0)
                    block = "narr"
                elif line.startswith("</top>") or line.startswith("<top>"):
                    block = None
                elif line.startswith("</title>") or line.startswith("</desc>") or line.startswith("</narr>"):
                    block = None
                elif block == "title":
                    title[qid].extend(line.strip().split())
                    if len(title[qid])>0:
                        if title[qid][0] == 'Topic:':
                            title[qid].pop(0)
                elif block == "desc":
                    desc[qid].extend(line.strip().split())
                    if len(desc[qid])>0:
                        if desc[qid][0] == 'Description:':
                            desc[qid].pop(0)
                elif block == "narr":
                    narr[qid].extend(line.strip().split())
                    if len(narr[qid])>0:
                        if narr[qid][0] == 'Narrative:':
                            narr[qid].pop(0)
        out = {}
        if len(title) > 0:
            out["title"] = {qid: " ".join(terms) for qid, terms in title.items()}
        if len(desc) > 0:
            out["desc"] = {qid: " ".join(terms) for qid, terms in desc.items()}
        if len(narr) > 0:
            out["narr"] = {qid: " ".join(terms) for qid, terms in narr.items()}

        return out

    def get_prep(
        self,
        plen = 150, 
        overlap = 50, 
        tlen = 0,
        max_pass_per_doc = 30,
    ):
        return TRECDocumentPrepFromRetriever(self.how, plen, overlap, tlen, max_pass_per_doc)
    

class Robust04(TRECCollection):

    def __init__(self, how='tokens'):
        super().__init__('Robust04')
        self.how = how.lower()

    #@clean_output_text
    def parse_doc(self, content, use_doc_title = False):
        ls = content.split('\n')
        see_text = False
        see_title = False
        doc = ''
        title = ''
        for l in ls:
            l = l.replace('\n', '').strip()
            if '<TEXT>' in l:
                see_text = True
            elif '</TEXT>' in l:
                break
            elif see_text:
                if l == '<P>' or l == '</P>':
                    continue
                doc += l + ' '
            elif '<HEADLINE>' in l:
                see_title =True
            elif '</HEADLINE>' in l :
                see_title = False
            elif see_title:
                if l == '<P>' or l == '</P>':
                    continue
                title += l + ' '
        title = title.strip() 
        content = doc.strip()
        content = re.sub(r'----*', '---', content)
        content = re.sub(r'  *', ' ', content)
        content = re.sub(r"\t", " ", content)
        content = re.sub(r"\n", " ", content)
        return title, content

class Core17(TRECCollection):

    def __init__(self, how='tokens'):
        super().__init__('Core17')
        self.how = how.lower()

    @clean_output_text
    def parse_doc(self, content, use_doc_title = False):
        """
        This parsing needs to be curated not all documents have titles
        """
        title = ''
        if use_doc_title:
            paraphs = content.split('\n', 1)
            if len(paraphs)>1 :
                title = paraphs[0]
                content = paraphs[1]

        return title, content

class Core18(TRECCollection):

    def __init__(self, how='tokens'):
        super().__init__('Core18')
        self.how = how.lower()

    @clean_output_text
    def parse_doc(self, content, use_doc_title = False):
        content_json = json.loads(content)
        content = ''
        title = ''
        for each in content_json['contents']:
            if each is not None and 'content' in each.keys():
                if use_doc_title:
                    if each['type'] =='title':
                        title = each['content']
                    elif each['type']  not in ('kicker','byline','date'):
                        content += '{}\n'.format(each['content'])
                else:
                    content += '{}\n'.format(each['content'])
        return title, content

class Gov2(TRECCollection):

    def __init__(self, how='tokens'):
        super().__init__('Gov2')
        self.how = how.lower()

    #@clean_output_text  # check data_utils for preprocessing code if you need
    def parse_doc(self, content, use_doc_title = False):
        title = ''
        content = re.sub(r'----*', '---', content)
        content = re.sub(r'  *', ' ', content)
        content = re.sub(r"\t", " ", content)
        content = re.sub(r"\n", " ", content)
        return title, content

class ClueWeb(TRECCollection):

    def __init__(self, how='tokens'):
        super().__init__('ClueWeb')
        self.how = how.lower()

    @clean_output_text  
    def parse_doc(self, content, use_doc_title = False):
        title = ''
        # # The documents are not well formed html documents ==> bs4 can't work
        # soup = BeautifulSoup(content, "lxml")
        # if use_doc_title:
        #     title_tag = soup.html.head.title
        #     title = '' if title_tag is None else title_tag.text
        #     content = soup.html.body.text
        # else:
        #     content = soup.html.text
        return title, content
    
class MsMarcoDocument(DocumentCol):
    def __init__(self, how='tokens'):
        super().__init__('MsMarco-Doc')
        self.how = how.lower()
    
    def get_prep(
        self,
        plen = 150, 
        overlap = 50, 
        tlen = 0,
        max_pass_per_doc = 30,
    ):
        return MsMarcoDocumentPrep(self.how, plen, overlap, tlen, max_pass_per_doc)


COLLECTIONS={
    'msmarco' : MsMarcoPassage,
    'robust04' : Robust04,
    'core17' : Core17,
    'core18' : Core18,
    'gov2' : Gov2,
    'terabyte04.701-750' : Gov2,
    'terabyte05.751-800' : Gov2,
    'terabyte06.801-850' : Gov2,
    'cw09b' : ClueWeb,
    'msmarco-doc': MsMarcoDocument,
}