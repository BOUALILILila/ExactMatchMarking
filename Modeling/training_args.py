from dataclasses import dataclass, field

import os
import tensorflow as tf
from typing import Optional
from transformers import TFTrainingArguments

from absl import logging as logger

logger.set_verbosity(logger.INFO)


@dataclass
class CustomTFTrainingArguments(TFTrainingArguments):

    warmup_prop: Optional[float] = field(default=0.0, metadata={"help": "Linear warmup porportion of total training steps."})

    patience: Optional[int] = field(
        default=20, metadata={"help": "Patience for early stopping."},
    )
    best_score: Optional[int] = field(
        default=0, metadata={"help": "Best score if continue training with early stopping from ckpt."},
    )
    delta: Optional[float] = field(
        default=0., metadata={"help": "delta from best_score early stopping."},
    )
    do_early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "Do early stopping."},
    )
    ckpt_name: Optional[str] = field(
        default=None, metadata={"help": "Name of the directory containing all checkpoints."}
    )
    ckpt_dir: Optional[str] = field(
        default=None, metadata={"help": "Saving directory of the transformers checkpoints."}
    )
    max_ckpt_keep: Optional[int] = field(
        default=3, metadata={"help": "Max checkpoints to keep by the checkpoint manager."},
    )
    save_all_ckpts: Optional[bool] = field(
        default=False, metadata={"help": "Whether to save all transformer checkpoints or just the last."},
    )
    eval_all_checkpoints:Optional[bool] = field(
        default=False, metadata={"help": "Saving directory of the transformers checkpoints."}
    )
    
    def __post_init__(self):
        if self.ckpt_dir is None and self.ckpt_name is not None:
            self.ckpt_dir = os.path.join(self.output_dir, f'ckpt_{self.ckpt_name}')
        elif self.ckpt_dir is None and self.ckpt_name is None:
            raise ValueError('TrainingArguments: No checkpoint specified !')
