"""Tensorflow trainer class."""
""" This class is mostly copied from huggingface's transformers library v3.0.2"""

import typing
import pandas as pd
import numpy as np
import os
import math
import random
from typing import Callable, Dict, Optional, Tuple
from tqdm import  trange
from tqdm import tqdm
import tensorflow as tf 
from transformers import (
                            TFPreTrainedModel,
                            GradientAccumulator, 
                            create_optimizer, 
                        )

from absl import logging as logger

from .metrics import BaseMetric
from .training_args import CustomTFTrainingArguments
from .trainer_utils import EarlyStopping



logger.set_verbosity(logger.INFO)


class CustomTFTrainer:
    model: TFPreTrainedModel
    args: CustomTFTrainingArguments
    train_dataset: Optional[tf.data.Dataset]
    num_train_examples: Optional[int]
    eval_dataset: Optional[tf.data.Dataset]
    num_eval_examples: Optional[int] 
    eval_set_name: Optional[str] = None
    query_doc_ids: Optional[pd.DataFrame] = None
    eval_qrels: Optional[pd.DataFrame] = None
    eval_metric: Optional[BaseMetric] = None
    compute_metrics: Optional[Callable[[typing.Any, typing.Union[list, np.array], typing.Union[list, np.array]],float]] = None
    prediction_loss_only: bool
    tb_writer: Optional[tf.summary.SummaryWriter] = None
    optimizers: Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.schedules.LearningRateSchedule] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None

    def __init__(
        self,
        model: TFPreTrainedModel,
        args: CustomTFTrainingArguments,
        train_dataset: Optional[tf.data.Dataset] = None,
        num_train_examples: Optional[int] = None,
        eval_dataset: Optional[tf.data.Dataset] = None,
        num_eval_examples: Optional[int] = None,
        out_suffix: Optional[str] = None,
        eval_metric: Optional[BaseMetric] = None,
        compute_metrics: Optional[Callable[[typing.Any, typing.Union[list, np.array], typing.Union[list, np.array]],float]] = None, 
        query_doc_ids: Optional[pd.DataFrame] = None,
        eval_qrels: Optional[pd.DataFrame] = None,
        prediction_loss_only=False,
        tb_writer: Optional[tf.summary.SummaryWriter] = None,
        optimizers: Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.schedules.LearningRateSchedule] = None,
    ):
        self.model = model
        self.strategy = args.strategy
        self.args = args
        self.train_dataset = train_dataset
        self.num_train_examples = num_train_examples
        self.eval_dataset = eval_dataset
        self.num_eval_examples = num_eval_examples
        self.out_suffix = out_suffix
        self.query_doc_ids = query_doc_ids
        self.eval_qrels = eval_qrels
        self.eval_metric = eval_metric
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        self.gradient_accumulator = GradientAccumulator()
        self.optimizer = None
        
        self.set_seed()

        if tb_writer is not None:
            self.tb_writer = tb_writer
        else:
            self.tb_writer = tf.summary.create_file_writer(self.args.logging_dir)

    def get_train_tfdataset(self) -> tf.data.Dataset:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        self.total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps

        if self.num_train_examples < 0:
            raise ValueError("The training dataset must have an asserted cardinality")
        # if self.args.max_steps > 0:
        #     self.train_steps = self.args.max_steps
        # else:
        #     self.train_steps: int = math.ceil(self.num_train_examples / self.args.train_batch_size) # math.floor because drop last batch

        return self.strategy.experimental_distribute_dataset(self.train_dataset)

    def get_eval_tfdataset(self, eval_dataset: Optional[tf.data.Dataset] = None,
                           num_eval_examples: Optional[int] = None, 
                           query_doc_ids: Optional[pd.DataFrame] = None,
                           eval_qrels: Optional[pd.DataFrame] = None,
                           prediction_loss_only: Optional[bool]= None) -> Tuple[tf.data.Dataset,
                                                                                int, pd.DataFrame,
                                                                                pd.DataFrame, bool]:

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        num_eval_examples = num_eval_examples if num_eval_examples is not None else self.num_eval_examples

        if num_eval_examples is None or (num_eval_examples is not None and num_eval_examples<=0):
            raise ValueError("Trainer: evaluation requires a non-empty dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        query_doc_ids = query_doc_ids if query_doc_ids is not None else self.query_doc_ids

        if query_doc_ids is None :
            raise ValueError("Trainer: evaluation requires an query_doc_ids.")

        eval_qrels = eval_qrels if eval_qrels is not None else self.eval_qrels

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        if (not prediction_loss_only or self.args.do_early_stopping) and (query_doc_ids is None or eval_qrels is None):
            raise ValueError("Trainer: Evaluation metrics caluculation requires query_doc_ids and eval_qrels.")

        return (self.strategy.experimental_distribute_dataset(eval_dataset), 
                num_eval_examples, query_doc_ids, eval_qrels, prediction_loss_only)

    def get_test_tfdataset(self, test_dataset: tf.data.Dataset, 
                            num_test_examples: int,
                            query_doc_ids: pd.DataFrame,
                           ) -> tf.data.Dataset:

        if test_dataset is None or query_doc_ids is None:
            raise ValueError("Trainer: test requires a test dataset and ids file.")
        if num_test_examples<=0:
            raise ValueError("Trainer: test set is empty.")

        return self.strategy.experimental_distribute_dataset(test_dataset)

    def get_optimizers(
        self,
    ) -> Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.schedules.LearningRateSchedule]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers

        optimizer, scheduler = create_optimizer(
            self.args.learning_rate,
            self.train_steps,
            self.args.warmup_steps,
            adam_epsilon=self.args.adam_epsilon,
            weight_decay_rate=self.args.weight_decay,
        )
        self.optimizers = (optimizer, scheduler)
        self.optimizer = optimizer

    def get_early_stopping(self)-> EarlyStopping:
        if self.args.patience is None or self.compute_metrics is None:
            raise ValueError("Trainer: Early stopping requires patience and eval metric.")
        return EarlyStopping(patience = self.args.patience, best_score= self.args.best_score,
                 save_dir = f'{self.args.ckpt_dir}', ckpt_manager = self.model.ckpt_manager )

    @tf.function
    def _evaluate_steps(self, per_replica_features, per_replica_labels):
        """
        One step evaluation across replica.
        Args:
          per_replica_features: the batched features.
          per_replica_labels: the batched labels.
        Returns:
          The loss corresponding to the given batch.
        """
        per_replica_loss, per_replica_logits, per_replica_labels, per_replica_ids = self.strategy.run(
            self._run_model, args=(per_replica_features, per_replica_labels)
        )

        try:
            reduced_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=0)
        except ValueError:
            reduced_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)

        return reduced_loss, per_replica_logits, per_replica_labels, per_replica_ids

    def _prediction_loop(
        self, dataset: tf.data.Dataset, num_examples:int, description: str, prediction_loss_only: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        # label_ids: np.ndarray = None
        preds: np.ndarray = None
        pairids: np.ndarray = None

        step: int = 1

        num_steps = (
            math.ceil(num_examples / self.args.eval_batch_size)
        )

        eval_iterator = tqdm(dataset, total=num_steps, 
                                   desc="Iteration")
        
        for features, labels_ in eval_iterator:
            step = tf.convert_to_tensor(step, dtype=tf.int64)
            loss, logits, labels, ids = self._evaluate_steps(features, labels_)
            loss = tf.reduce_mean(loss)

            if not prediction_loss_only:
                # if isinstance(logits, tuple):
                #     logits = logits[0]

                # if isinstance(labels, tuple):
                #     labels = labels[0]
                
                # if isinstance(ids, tuple):
                #     ids = ids[0]

                if self.args.n_gpu > 1:
                    for val in logits.values:
                        if preds is None:
                            preds = val.numpy()
                        else:
                            preds = np.append(preds, val.numpy(), axis=0)
                    
                    for val in ids.values:
                        if pairids is None:
                            pairids = val.numpy()
                        else:
                            pairids = np.append(pairids, val.numpy(), axis=0)

                    # for val in labels.values:
                    #     if label_ids is None:
                    #         label_ids = val.numpy()
                    #     else:
                    #         label_ids = np.append(label_ids, val.numpy(), axis=0)
                else:
                    if preds is None:
                        preds = logits.numpy()
                    else:
                        preds = np.append(preds, logits.numpy(), axis=0)

                    if pairids is None:
                        pairids = ids.numpy()
                    else:
                        pairids = np.append(pairids, ids.numpy(), axis=0)

                    # if label_ids is None:
                    #     label_ids = labels.numpy()
                    # else:
                    #     label_ids = np.append(label_ids, labels.numpy(), axis=0)

            step += 1
        
        # end of prediction loop
        dict_preds = {'id': pairids, 'logits_0': preds[:,0], 'logits_1': preds[:,1]}

        df_preds = pd.DataFrame.from_dict(dict_preds).astype({'id': str, 'logits_0': float, 'logits_1': float})

        return loss.numpy() , df_preds

    def evaluate(
        self, eval_dataset: Optional[tf.data.Dataset] = None,  
        prediction_loss_only: Optional[bool] = None, step = None,
        num_eval_examples: Optional[int] = None, 
        query_doc_ids: Optional[pd.DataFrame] = None,
        eval_qrels: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.
        """
        
        eval_ds, num_eval_examples, query_doc_ids, eval_qrels, prediction_loss_only = self.get_eval_tfdataset(eval_dataset, 
                                                                                        num_eval_examples,
                                                                                        query_doc_ids,
                                                                                        eval_qrels,
                                                                                        prediction_loss_only)

        # out_file
        file_name = f"{step}" if self.out_suffix=='' else f"{self.out_suffix}_{step}"
        preds_file = tf.io.gfile.GFile(os.path.join(self.args.output_dir, f'predictions_{file_name}_all.tsv'), 'w')

        loss, df_preds = self._prediction_loop(eval_ds, num_eval_examples, description="Evaluation")


        preds_with_ids = pd.merge(df_preds,query_doc_ids, on='id')  
        
        for i, row in preds_with_ids.iterrows():
              preds_file.write("\t".join((str(row['qid']), str(row['did']), str(row['pass']), str(row['logits_0']), str(row['logits_1']))) + "\n")

        preds_file.close()

        
        metrics ={}
        if not prediction_loss_only:
            df = preds_with_ids.sort_values(by=['qid','did','logits_1'], ascending=[True,True,False])
            df['pred_max']=df.groupby(['qid','did'], sort=False)['logits_1'].transform(max)
            df= df.drop_duplicates(['qid','did'], keep='first')
            df = df.sort_values(by=['qid','pred_max'], ascending=[True, False])
            df_preds_rel = pd.merge(df, eval_qrels, on=['qid','did'], how ='left')
            df_preds_rel['label'] = df_preds_rel['label'].fillna(0)

            if self.compute_metrics is not None:
                metrics[repr(self.eval_metric)] = self.compute_metrics(df_preds_rel['qid'].values, 
                                                                df_preds_rel['label'].values, 
                                                                df_preds_rel['pred_max'].values)
            elif self.args.do_early_stopping:
                raise ValueError("Trainer: Early stopping requires compute metrics.")
            else:
                logger.info('No metric was given for evaluation.')

        metrics["eval_loss"] = loss

        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return metrics

    def train(self) -> None:
        """
        Train method to train the model.
        """
        train_ds = self.get_train_tfdataset()

        if self.args.debug:
            tf.summary.trace_on(graph=True, profiler=True)

        self.gradient_accumulator.reset()

        # here actualy the concept of epoch does not really affect the computation
        # It s the same when using max_steps = (num_train_examples / batch_size)*epochs with epochs=1
        # plus the iterator is saved so we restor at the right batch 

        num_update_steps_per_epoch = self.num_train_examples / self.total_train_batch_size

        # here we drop the last incomplete batch so we use math.floor instead of ceil
        approx = math.floor 
        num_update_steps_per_epoch = approx(num_update_steps_per_epoch)

        # At least one update for each epoch.
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        self.steps_per_epoch = num_update_steps_per_epoch

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            epochs = (self.args.max_steps // self.steps_per_epoch) + int(
                self.args.max_steps % self.steps_per_epoch > 0
            )
        else:
            t_total = self.steps_per_epoch * self.args.num_train_epochs
            epochs = self.args.num_train_epochs

        # Since ``self.args.num_train_epochs`` can be `float`, make ``epochs`` be a `float` always.
        epochs = float(epochs)

        with self.strategy.scope():
            self.get_optimizers()
            iterator = iter(train_ds)
            ckpt = tf.train.Checkpoint(optimizer = self.optimizer, model = self.model, iterator = iterator)
            self.model.ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                                 os.path.join(self.args.output_dir,f'tf_ckpt_{self.args.ckpt_name}'),
                                                                 max_to_keep=self.args.max_ckpt_keep) 
            
            iterations = self.optimizer.iterations
            epochs_trained = 0
            steps_trained_in_current_epoch = 0

            if self.model.ckpt_manager.latest_checkpoint:
                logger.info(
                    "Checkpoint file %s found and restoring from checkpoint", self.model.ckpt_manager.latest_checkpoint
                )

                ckpt.restore(self.model.ckpt_manager.latest_checkpoint).expect_partial()

                step = iterations.numpy()

                epochs_trained = step // self.steps_per_epoch
                steps_trained_in_current_epoch = step % self.steps_per_epoch

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)


        tf.summary.experimental.set_step(iterations)

        if self.args.fp16:
            policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
            tf.keras.mixed_precision.experimental.set_policy(policy)

        with self.tb_writer.as_default():
            tf.summary.text("args", self.args.to_json_string())

        self.tb_writer.flush()

        if self.args.do_early_stopping:
            early_stopping = self.get_early_stopping()
            self.args.save_steps = float('inf')

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_train_examples)
        logger.info("  Num Epochs = %d", epochs)
        logger.info("  Total optimization steps = %d", self.train_steps)

        for epoch in range(epochs_trained, int(epochs)):
            logger.info("Starting Epoch {} ...".format(epoch+1))
            for training_loss in self._training_steps(iterator): # do train on batch, apply gradients return loss
                step = iterations.numpy()
                if self.args.debug:
                    with self.tb_writer.as_default():
                        tf.summary.scalar("loss", training_loss, step=step)

                if step == 1 and self.args.debug:
                    with self.tb_writer.as_default():
                        tf.summary.trace_export(name="training", step=step, profiler_outdir=self.args.logging_dir)

                if (self.args.evaluate_during_training or self.args.do_early_stopping) and step % self.args.eval_steps == 0: # eval_during _training dev set
                    logs = {}
                    results = self.evaluate(step=step)

                    for key, value in results.items():
                        eval_key = "eval_{}".format(key)
                        logs[eval_key] = value

                    logs["learning_rate"] = self.optimizers[1](step).numpy()

                    logger.info("Epoch {} Step {} Validation Metrics {}".format(epoch, step, logs))

                    with self.tb_writer.as_default():
                        for k, v in logs.items():
                            tf.summary.scalar(k, v, step=step)

                    self.tb_writer.flush()
                    
                    if self.args.do_early_stopping:
                        early_stopping(results[f'eval_{repr(self.eval_metric)}'], self.model, step)  ## self.args.eval_metric

                if step % self.args.logging_steps == 0:
                    logger.info("Epoch {} Step {} Train Loss {:.4f}".format(epoch, step, training_loss.numpy()))

                if step % self.args.save_steps == 0:
                    # 1: save tf trainable ckpt 
                    ckpt_save_path = self.model.ckpt_manager.save()
                    logger.info("Saving checkpoint for step {} at {}".format(step, ckpt_save_path))

                    if self.args.save_all_ckpts :
                        # 2: save transformer ckpt
                        save_chkpt_dir = os.path.join(self.args.ckpt_dir, f"checkpoint-{step}")
                        self.save_model(save_chkpt_dir)
                
                if self.args.do_early_stopping and early_stopping.early_stop:
                    break      

                if self.args.max_steps > 0 and step >= t_total :
                    break

                if step % self.steps_per_epoch == 0:
                    break
            
            if self.args.max_steps > 0 and step >= self.args.max_steps:
                break
                

    def _training_steps(self, iterator):
        """
        Returns a generator over training steps (i.e. parameters update).
        """
        for i, loss in enumerate(self._accumulate_next_gradients(iterator)):
            if i % self.args.gradient_accumulation_steps == 0:
                self._apply_gradients()
                yield loss

    @tf.function
    def _apply_gradients(self):
        """Applies the gradients (cross-replica)."""
        self.strategy.run(self._step)

    def _step(self):
        """Applies gradients and resets accumulation."""
        gradient_scale = self.gradient_accumulator.step * self.strategy.num_replicas_in_sync
        gradients = [
            gradient / tf.cast(gradient_scale, gradient.dtype) for gradient in self.gradient_accumulator.gradients
        ]
        gradients = [(tf.clip_by_value(grad, -self.args.max_grad_norm, self.args.max_grad_norm)) for grad in gradients]
        

        self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)))
        self.gradient_accumulator.reset()

    def _accumulate_next_gradients(self, iterator):
        """Accumulates the gradients from the next element in dataset."""

        @tf.function
        def _accumulate_next():
            per_replica_features, per_replica_labels = iterator.get_next_as_optional()

            return self._accumulate_gradients(per_replica_features, per_replica_labels)

        while True:
            try:
                yield _accumulate_next()
            except tf.errors.OutOfRangeError:
                break

    def _accumulate_gradients(self, per_replica_features, per_replica_labels):
        """Accumulates the gradients across all the replica."""
        per_replica_loss = self.strategy.run(
            self._forward, args=(per_replica_features, per_replica_labels)
        )

        try:
            reduced_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=0)
        except ValueError:
            reduced_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)

        return reduced_loss

    def _forward(self, features, labels):
        """Forwards a training example and accumulates the gradients."""
        per_example_loss, _ = self._run_model(features, labels, True) 
        gradients = tf.gradients(per_example_loss, self.model.trainable_variables)
        gradients = [
            g if g is not None else tf.zeros_like(v) for g, v in zip(gradients, self.model.trainable_variables)
        ]

        self.gradient_accumulator(gradients)

        return per_example_loss

    def _run_model(self, features, labels, training=False):
        """
        Computes the loss of the given features and labels pair.
        Args:
          features: the batched features.
          labels: the batched labels.
          training: run the model in training mode or not
        """
        bert_keys = ['input_ids', 'attention_mask','token_type_ids']
        bert_features = dict((k, features[k]) for k in bert_keys if k in features)

        if isinstance(labels, (dict)):
            loss, logits = self.model(bert_features, training=training, **labels)[:2]
        else:
            loss, logits = self.model(bert_features, labels=labels, training=training)[:2]
        loss += sum(self.model.losses) * (1.0 / self.args.n_gpu)
        
        outputs = (loss, logits,)

        if not training:
            outputs = outputs + (labels, features['id'])

        return outputs


    def predict(self, test_dataset: tf.data.Dataset, num_test_examples: int, query_doc_ids: Optional[pd.DataFrame], 
                test_qrels: Optional[pd.DataFrame], test_set_name:str) -> dict:
        """
        Run prediction and return predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        Args:
          test_dataset: something similar to a PT Dataset. This is just
            temporary before to have a framework-agnostic approach for datasets.
        """
        test_ds = self.get_test_tfdataset(test_dataset,
                                                        num_test_examples,
                                                        query_doc_ids)

        # out_files
        all_preds_file = tf.io.gfile.GFile(os.path.join(self.args.output_dir, f'predictions_{test_set_name}_all.tsv'), 'w')
        preds_file = tf.io.gfile.GFile(os.path.join(self.args.output_dir, f'predictions_{test_set_name}_maxP.tsv'), 'w')

        loss, df_preds = self._prediction_loop(test_ds, num_test_examples, description="Prediction")

        preds_with_ids = pd.merge(df_preds, query_doc_ids, on='id') 
        
        for i, row in preds_with_ids.iterrows():
            all_preds_file.write("\t".join((str(row['qid']), str(row['did']), str(row['pass']), str(row['logits_0']), str(row['logits_1']))) + "\n")

        all_preds_file.close()

        df = preds_with_ids.sort_values(by=['qid','did','logits_1'], ascending=[True,True,False])
        df['pred_max']=df.groupby(['qid','did'], sort=False)['logits_1'].transform(max)
        df= df.drop_duplicates(['qid','did'], keep='first')
        df = df.sort_values(by=['qid','pred_max'], ascending=[True, False])

        # df_preds_rel = pd.merge(df, test_qrels, on=['qid','did'], how ='left')
        # df_preds_rel['label'] = df_preds_rel['label'].fillna(0)

        # if self.compute_metrics is not None and preds is not None and label_ids is not None:
        #     metrics[self.eval_metric] = self.compute_metrics(df_preds_rel['qid'].values, 
        #                                                      df_preds_rel['label'].values, 
        #                                                      df_preds_rel['pred'].values)
        # else:
        #     metrics = {}

        # metrics["eval_loss"] = loss.numpy()

        # for key in list(metrics.keys()):
        #     if not key.startswith("eval_"):
        #         metrics[f"eval_{key}"] = metrics.pop(key)

        for i, row in df.iterrows():
              preds_file.write("\t".join((str(row['qid']), str(row['did']), str(row['pred_max']))) + "\n")

        preds_file.close()

        return dict()
    
    def set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        tf.random.set_seed(self.args.seed)

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.
        if outout_dir is given it is used to save the model else :
            if ckpt_dir is given in training params it is used else :
                use the default args.output_dir/args.ckpt_name
        """
        output_dir = output_dir if output_dir is not None else self.args.ckpt_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        logger.info("Saving model in {}".format(output_dir))

        self.model.save_pretrained(output_dir)

    def save_last_tf_chkpt(self, save_dir, ckpt_dir=None):
        with self.strategy.scope():
            # self.get_optimizers()
            # iterations = self.optimizer.iterations
            ckpt = tf.train.Checkpoint(model=self.model)
            dir_ckpt = ckpt_dir if ckpt_dir is not None else os.path.join(self.args.output_dir, f'tf_ckpt_{self.args.ckpt_name}')
            self.model.ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                                 dir_ckpt,
                                                                 max_to_keep=5) 

            if self.model.ckpt_manager.latest_checkpoint:
                logger.info(
                    "Checkpoint file %s found and restoring from checkpoint", self.model.ckpt_manager.latest_checkpoint
                )

                ckpt.restore(self.model.ckpt_manager.latest_checkpoint).expect_partial()
                self.save_model(save_dir)
            
            else :
                logger.info("No checkpoint found!")
