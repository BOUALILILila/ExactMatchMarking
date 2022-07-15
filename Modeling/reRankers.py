

 
import tensorflow as tf
from transformers import (
    TFAutoModelForSequenceClassification, 
    TFElectraForSequenceClassification, 
    TFBertForSequenceClassification,
    BatchEncoding,
    )
from transformers.modeling_tf_bert import TFBertMainLayer
from transformers.modeling_tf_outputs import TFSequenceClassifierOutput

from transformers.modeling_tf_electra import TFElectraMainLayer
from transformers.modeling_tf_utils import get_initializer

from absl import logging as logger
logger.set_verbosity(logger.INFO)

class TFElectraForRelevanceClassification(TFElectraForSequenceClassification):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.electra = TFElectraMainLayer(config, name="electra")
        self.classifier = TFElectraRelevanceHead(None, None, config, name="classifier")


class TFElectraRelevanceHead(tf.keras.layers.Layer):
    """ BERT-style ClassificationHead (i.e., out_proj only -- no dense). See transformers.TFElectraClassificationHead
        from https://github.com/capreolus-ir/capreolus/blob/master/capreolus/reranker/TFBERTMaxP.py """

    def __init__(self, dropout=None, out_proj=None, config=None, *args, **kwargs):
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


# New variants
MARKER_ID=1001

def get_pooler(config, add_cls: int = 0, pooling_method: str = 'avg'):
    if pooling_method.lower() == 'avg':
        return AvgPooler(config, add_cls)
    elif pooling_method.lower() == 'first':
        return FirstPooler(config, add_cls)
    elif pooling_method.lower() == 'max':
        return MaxPooler(config, add_cls)
    else:
        raise ValueError(f"Unknown pooling method = {pooling_method}")

class MarkerPooler(tf.keras.layers.Layer):
    def __init__(self, config, add_cls: int = 0):
        super().__init__()
        self.add_cls = add_cls
        n = 3 if (self.add_cls == 1) else 2 
        logger.info(f"dense n = {n}")
        self.dense = tf.keras.layers.Dense(
            config.hidden_size*n,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )

    def call(self, sequence_output, q_mask, d_mask):
        pooled_q = self.pool(sequence_output, q_mask) # (bs, D)
        pooled_d = self.pool(sequence_output, d_mask) # (bs, D)
        if self.add_cls ==  1:
            cls_reps = sequence_output[:, 0] # (bs, D)
            pooled_output = tf.concat([cls_reps, pooled_q, pooled_d], axis=-1) # (bs, D*3)
        else:
            pooled_output = tf.concat([pooled_q, pooled_d], axis=-1) # (bs, D*2)
        pooled_output = self.dense(pooled_output)
        return pooled_output
    
    def pool(self, marker_reps):
        raise NotImplementedError()

class AvgPooler(MarkerPooler):
    def pool(self, sequence_reps, marker_mask):
        marker_reps = sequence_reps * tf.expand_dims(tf.cast(marker_mask, sequence_reps.dtype), axis=-1)
        sum_reps =  tf.math.reduce_sum(marker_reps, axis=1)
        n = tf.clip_by_value(
                    tf.math.reduce_sum(tf.cast(marker_mask, marker_reps.dtype), axis=1), 
                    clip_value_min=1e-9, clip_value_max=512 # avoid division by 0
                ) 
        return sum_reps / tf.expand_dims(n, axis=-1)

class FirstPooler(MarkerPooler):
    def pool(self, sequence_reps, marker_mask):
        # build first mask
        first_marker_indices = tf.argmax(marker_mask, axis=1, output_type=marker_mask.dtype)
        indices = tf.stack([tf.range(tf.shape(first_marker_indices)[0]), first_marker_indices], axis=-1) # 2-D indices (add batch dim)

        updates = tf.repeat([1], repeats=[tf.shape(indices)[0]])

        mat = tf.zeros_like(marker_mask)
        scatter = tf.tensor_scatter_nd_update(mat, indices, updates)
        
        mask = marker_mask * scatter # argmax on all 0s is the first 0 we need to get rid of it

        return tf.math.reduce_max(sequence_reps * tf.expand_dims(tf.cast(mask, sequence_reps.dtype), axis=-1), axis=1)

class MaxPooler(MarkerPooler):
    def pool(self, sequence_reps, marker_mask):
        marker_reps = sequence_reps * tf.expand_dims(tf.cast(marker_mask, sequence_reps.dtype), axis=-1)
        return tf.math.reduce_max(marker_reps, axis=1)

class TFBertForRelevanceClassificationWithPooling(TFBertForSequenceClassification):
    def __init__(self, config, *inputs, **kwargs):
        pooling_method = kwargs.pop('pooling_method','avg')
        self.add_cls = kwargs.pop('add_cls', 0)

        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.bert = TFBertMainLayer(config, name="bert")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )

        self.pooler = get_pooler(config = config, add_cls=self.add_cls, pooling_method=pooling_method)
        logger.info(f"Using Pooling method : {pooling_method}.")
        logger.info(f"add_cls = {self.add_cls}")
    
    @property
    def dummy_inputs(self):
        """Dummy inputs to build the network.
        Returns:
            tf.Tensor with dummy inputs
        """
        DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
        DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]
        DUMMY_SEG = [[0,0,1,1,1],[0,0,0,1,1],[0,1,1,1,1]]
        return {
            "input_ids": tf.constant(DUMMY_INPUTS), 
            "attention_mask": tf.constant(DUMMY_MASK), 
            "token_type_ids": tf.constant(DUMMY_SEG)
        }

    def call(
        self,
        inputs=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        training=False,
    ):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.bert.return_dict

        if isinstance(inputs, (tuple, list)):
            labels = inputs[9] if len(inputs) > 9 else labels
            if len(inputs) > 9:
                inputs = inputs[:9]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop("labels", labels)

        outputs = self.bert(
            inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = outputs[0]

        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            token_type_ids = inputs[2] if len(inputs) > 2 else token_type_ids
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
        else:
            input_ids = inputs
        
        # marker tokens mask
        markers_mask = (input_ids == MARKER_ID)
        markers_mask = tf.cast(markers_mask, token_type_ids.dtype)
        q_mask = (1-token_type_ids) * markers_mask
        d_mask = token_type_ids * markers_mask

        pooled_output = self.pooler(sequence_output, q_mask, d_mask)

        if self.add_cls == 2:
            cls_pooler = outputs[1]
            pooled_output = tf.concat([cls_pooler, pooled_output], axis=-1) # (bs, D*3)
            assert pooled_output.shape == [input_ids.shape[0], sequence_output.shape[-1]*3], f"pooled_output.shape={pooled_output}"

        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        loss = None if labels is None else self.compute_loss(labels, logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )