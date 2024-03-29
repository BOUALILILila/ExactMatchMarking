# coding=utf-8

from absl import logging as logger
import os, glob, re
import pandas as pd
import tensorflow as tf
from transformers import (
    TF2_WEIGHTS_NAME,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    TFAutoModelForSequenceClassification,
    TFTrainingArguments,
)

from Modeling import (
    CustomTFTrainer,
    CustomTFTrainingArguments,
    get_eval_metric,
)
from args import (
    ModelArguments,
    DataTrainingArguments,
)


logger.set_verbosity(logger.INFO)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTFTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        tf.io.gfile.exists(training_args.output_dir)
        and tf.io.gfile.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    logger.info(
        "n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.n_gpu,
        bool(training_args.n_gpu > 1),
        training_args.fp16,
    )
    logger.info("Training/evaluation/prediction parameters %s", training_args)

    if training_args.do_train or training_args.do_eval:
       
        num_labels = 2
        output_mode = 'classification'

        # Load pretrained model and tokenizer
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
            output_hidden_states=False,
        )
        # tokenizer = AutoTokenizer.from_pretrained(
        #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        #     cache_dir=model_args.cache_dir,
        # )

        with training_args.strategy.scope():
            model = TFAutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_pt=True if glob.glob(f"{model_args.model_name_or_path}/*.bin") else False,
                config=config,
                cache_dir=model_args.cache_dir,
            )


        # Get datasets
        if training_args.do_train:
            filename = os.path.join(data_args.train_data_dir, f'dataset_train_{data_args.train_set_name}.tf')
            if tf.io.gfile.exists(filename):
                train_dataset, num_train_examples = data_args.doc_processor.get_train_dataset(filename, 
                                                                            training_args.train_batch_size, training_args.seed)
            else:
                raise IOError('File does not exist: ', filename)
        else:
            train_dataset, num_train_examples = None, 0

        if training_args.do_eval or training_args.do_early_stopping:
            filename = os.path.join(data_args.eval_data_dir, f'dataset_dev_{data_args.eval_set_name}.tf')
            ids_file = os.path.join(data_args.eval_data_dir, f'query_pass_ids_dev_{data_args.eval_set_name}.tsv')
            if tf.io.gfile.exists(filename):
                eval_dataset, num_eval_examples = data_args.doc_processor.get_eval_dataset(filename, 
                                                                        training_args.eval_batch_size)
            else:
                raise IOError('File does not exist: ', filename)
            if tf.io.gfile.exists(ids_file):
                query_doc_ids = pd.read_csv(ids_file, 
                                            header=None, index_col=None, delimiter='\t', 
                                            names=['id','qid','did','pass'], 
                                            dtype={'id':str, 'qid':str,'did':str})
            else:
                raise IOError('File does not exist: ', ids_file)

            if data_args.eval_qrels_file is not None:
                qrels_file = os.path.join(data_args.eval_data_dir, f'{data_args.eval_qrels_file}.tsv')
                if tf.io.gfile.exists(qrels_file):
                    eval_qrels = pd.read_csv(qrels_file,
                                header=None, index_col=None, delimiter=' ', names=['qid','_','did','rel'], 
                                dtype={'qid':str,'did':str, 'label':int})
                else:
                    raise IOError('File does not exist: ', qrels_file)
            else:
                eval_qrels = None
        else:
            eval_dataset, num_eval_examples, query_doc_ids, eval_qrels = None, 0, None, None

        eval_metric = get_eval_metric(data_args.collection)


        # Initialize our Trainer
        trainer = CustomTFTrainer(
            model = model,
            args = training_args,
            train_dataset = train_dataset,
            num_train_examples = num_train_examples,
            eval_dataset = eval_dataset,
            num_eval_examples = num_eval_examples,
            out_suffix = f'{data_args.eval_set_name}_{data_args.out_suffix}', # eval output file name
            eval_metric = eval_metric,
            compute_metrics = eval_metric.compute_on_df,
            query_doc_ids = query_doc_ids,
            eval_qrels = eval_qrels,
        )

        # Training
        if training_args.do_train:
            trainer.train()

            if not training_args.do_early_stopping:
                trainer.save_model()

        # Evaluation        
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            result = trainer.evaluate()
            output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{data_args.eval_set_name}.txt")

            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")

                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

                results.update(result)

    # Test
    if training_args.do_predict:

        checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(f'{training_args.ckpt_dir}' + "/**/" + TF2_WEIGHTS_NAME, recursive=True),
                    key=lambda f: int("".join(filter(str.isdigit, f)) or -1),
                )
            )

        if len(checkpoints) == 0:
            raise IOError('No checkpoint found at this location: ', training_args.ckpt_dir)

        elif not training_args.eval_all_checkpoints:
            if training_args.ckpt_dir in checkpoints:
                checkpoints = [training_args.ckpt_dir]
            else:
                checkpoints = [checkpoints[-1]]    
                    
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        filename = os.path.join(data_args.eval_data_dir, f'dataset_test_{data_args.test_set_name}.tf')
        if tf.io.gfile.exists(filename):
            test_dataset, num_test_examples = data_args.doc_processor.get_eval_dataset(filename, 
                                                    training_args.eval_batch_size)
        else: 
            raise IOError('File does not exist: ', filename)

        ids_file = os.path.join(data_args.eval_data_dir, f'query_pass_ids_test_{data_args.test_set_name}.tsv')
        if tf.io.gfile.exists(ids_file):
            query_doc_ids = pd.read_csv(ids_file,
                                header=None, index_col=None, delimiter='\t', 
                                names=['id','qid','did','pass'],
                                dtype={'id':str, 'qid':str,'did':str})
        else: 
            raise IOError('File does not exist: ', ids_file)

        if data_args.test_qrels_file:
            qrels_file = os.path.join(data_args.eval_data_dir, f'{data_args.test_qrels_file}.tsv')
            if tf.io.gfile.exists(qrels_file):
                test_qrels = pd.read_csv(qrels_file, 
                            header=None, index_col=None, delimiter='\t', 
                            names=['qid','did','label'], 
                            dtype={'qid':str,'did':str, 'label':int})
                
            else :
                raise IOError('File does not exist: ', qrels_file)
        else:
            test_qrels = None

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if re.match(".*checkpoint*-[0-9]", checkpoint) else "final"
            logger.info("Evaluate the following checkpoint-step: %s - %s", checkpoint, global_step)
            print("Evaluate the following checkpoint-step:", checkpoint, global_step)

            with training_args.strategy.scope():
                trained_model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
                trained_model.summary()
        
            trainer = CustomTFTrainer(
                          model = trained_model,
                          args = training_args,
                      )

            trainer.predict(test_dataset, num_test_examples, query_doc_ids, test_qrels, 
                            f'{data_args.test_set_name}_{data_args.out_suffix}-{global_step}')

if __name__ == "__main__":
    main()