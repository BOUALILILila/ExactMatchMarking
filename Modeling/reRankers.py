

 
import tensorflow as tf
from transformers import (
    TFAutoModelForSequenceClassification, 
    TFElectraForSequenceClassification, 
    TFBertForSequenceClassification
    )
from transformers.modeling_tf_electra import TFElectraMainLayer
from transformers.modeling_tf_utils import get_initializer

class TFElectraForRelevanceClassification(TFElectraForSequenceClassification):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.electra = TFElectraMainLayer(config, name="electra")
        self.classifier = TFElectraRelevanceHead(dropout=None, out_proj=None, config, name="classifier")


class TFElectraRelevanceHead(tf.keras.layers.Layer):
    """ BERT-style ClassificationHead (i.e., out_proj only -- no dense). See transformers.TFElectraClassificationHead
        from https://github.com/capreolus-ir/capreolus/blob/master/capreolus/reranker/TFBERTMaxP.py """

    def __init__(self, dropout=None, out_proj=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = dropout if dropout is not None else tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.out_proj = out_proj if out_proj is not None else tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )

    def call(self, inputs, **kwargs):
        x = inputs[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# class ClassificationHeadWrapper:

#     def __init__(self):
#         self.electra_classifier = None

#     def wrap_relevance_head(self, model):
#         if isinstance(model, TFElectraForSequenceClassification):
#             dropout, fc = model.classifier.dropout, model.classifier.out_proj
#             self.electra_classifier = model.classifier
#             model.classifier = TFElectraRelevanceHead(dropout, fc, name="classifier") 
#         return model

#     def unwrap_relevance_head(self, model):
#         if isinstance(model, TFElectraForSequenceClassification):
#             if self.electra_classifier is None :
#                 raise ValueError('Cannot unwarap relevance head, Electra Classification Head is None')
#             self.electra_classifier.dropout, self.electra_classifier.out_proj = model.classifier.dropout, model.classifier.out_proj
#             model.classifier = self.electra_classifier
#         return model

def wrap_relevance_head(model):
    if isinstance(model, TFElectraForSequenceClassification):
        dropout, fc = model.classifier.dropout, model.classifier.out_proj
        model.classifier = TFElectraRelevanceHead(dropout, fc, name="classifier") 
    return model
