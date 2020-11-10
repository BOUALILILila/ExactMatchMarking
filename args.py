from dataclasses import dataclass, field
from typing import Optional
from Data import (
    get_marker, 
    get_collection,
)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    collection: str =field(
        metadata={"help": "collection name"}
    )
    marking_strategy: str =field(
        metadata={"help": "base, sin_pair, pre_pair,...."}
    )
    train_data_dir:Optional[str] = field(
        default=None, metadata={"help": "The directory where the training data is."}
    )
    eval_data_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory containing the eval / dev sets."}
    )
    train_set_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the fine-tuning set."}
    )
    eval_set_name: Optional[str] = field(
        default=None, metadata={"help": "Eval set name."}
    )
    out_suffix: Optional[str] = field(
        default='', metadata={"help": "Predictions out file name suffix."}
    )
    test_set_name: Optional[str] =field(
        default=None, metadata={"help": "Test set name."}
    )
    eval_qrels_file: Optional[str] =field(
        default=None, metadata={"help": "Eval qrels filename."}
    )
    test_qrels_file: Optional[str] =field(
        default=None, metadata={"help": "Test qrels filename."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        }
    )
    # max_query_length: int = field(
    #     default=64,
    # )
    # max_title_length: int = field(
    #     default=64,
    # )
    # chunk_size: int = field(
    #     default=384,
    # )
    # stride: int = field(
    #     default=192,
    # )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        col = get_collection(self.collection)
        self.doc_processor = col.get_processor(self.max_seq_length)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    # tokenizer_name: Optional[str] = field(
    #     default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    # )
    # use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )