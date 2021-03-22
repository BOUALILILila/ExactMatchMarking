

 
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, TFElectraForSequenceClassification, TFBertForSequenceClassification


class TFElectraRelevanceHead(tf.keras.layers.Layer):
    """ BERT-style ClassificationHead (i.e., out_proj only -- no dense). See transformers.TFElectraClassificationHead
        from https://github.com/capreolus-ir/capreolus/blob/master/capreolus/reranker/TFBERTMaxP.py """

    def __init__(self, dropout, out_proj, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = dropout
        self.out_proj = out_proj

    def call(self, inputs, **kwargs):
        x = inputs[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def wrap_relevance_head(model):
    if isinstance(model, TFElectraForSequenceClassification):
        dropout, fc = model.classifier.dropout, model.classifier.out_proj
        model.classifier = TFElectraRelevanceHead(dropout, fc)
    return model