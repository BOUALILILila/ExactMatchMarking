from .marker_utils import get_marker
from .udel_stop_words import stopwords
from .convert_to_udel import UdelConverter


from .data_utils import TRECDocumentPrepFromRetriever, MsMarcoPassagePrep
from .io_handles import PassageHandle, DocumentSplitterHandle
from .data_processors import DocumentProcessor, PassageProcessor
from .collections import get_available_collections, get_collection, DocumentCol

# COLLECTIONS = ['msmarco', 'robust04', 'core17', 'core18']

# COLLECTION_PROCESSORS={
#     'msmarco' : (PassageProcessor, PassageHandle),
#     'robust04': (DocumentProcessor, DocumentSplitterHandle),
#     'core17' : (DocumentProcessor, DocumentSplitterHandle),
#     'core18' : (DocumentProcessor, DocumentSplitterHandle),
# }
# assert set(COLLECTION_PROCESSORS.keys()) == set(COLLECTIONS), 'DATA.init: Each collection must have defined processors'

# COLLECTION_PREPS={
#     'robust04' : TRECDocumentPrepFromRetriever,
#     'core17' : TRECDocumentPrepFromRetriever,
#     'core18' : TRECDocumentPrepFromRetriever,
#     'msmarco' : MsMarcoPassagePrep,
# }
# assert set(COLLECTION_PREPS.keys()) == set(COLLECTIONS), 'DATA.init: Each collection must have a defined prep'

# def get_collection_prep(collection):
#     collection = collection.lower()
#     if collection not in COLLECTION_PREPS.keys():
#         raise ValueError(f'Collection not recognized! It must be in {list(COLLECTION_PREPS.keys())}')
#     return  COLLECTION_PREPS[collection]

# def get_collection_processors(collection):
#     collection = collection.lower()
#     if collection not in COLLECTION_PROCESSORS.keys():
#         raise ValueError(f'Collection not recognized! It must be in {list(COLLECTION_PROCESSORS.keys())}')
#     return  COLLECTION_PROCESSORS[collection]

# def get_collection_names():
#     return COLLECTIONS

""" revise """
'''
mb collection needs to be added in data_utils
cord19 training needs to be added in data_utils
udel queries support for msmarco train add the argument in prep_data
'''