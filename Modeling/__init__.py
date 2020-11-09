
from .metrics import (  
                        BaseMetric,
                        DiscountedCumulativeGain, 
                        MeanReciprocalRank, 
                        MeanAveragePrecision, 
                        NormalizedDiscountedCumulativeGain, 
                        eval_metric_on_data_frame
                    )   
       
from .trainer_utils import EarlyStopping

from .training_args import CustomTFTrainingArguments
from .trainer import CustomTFTrainer

METRICS = {
    'msmarco' : MeanReciprocalRank(),
    'robust04' : MeanAveragePrecision(),
    'core17' : MeanAveragePrecision(),
    'core18' : MeanAveragePrecision(),
}

# from ..Data import get_collection_names

# assert set(METRICS.keys()) == set(get_collection_names()), 'Modeling.init: every collection must have a defined evaluation metric.'

def get_eval_metric(collection):
    collection = collection.lower()
    if collection not in METRICS.keys():
        raise ValueError(f'Collection not recognized! It must be in {list(METRICS.keys())}')
    return METRICS[collection]