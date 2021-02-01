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
    tf_ckpt_dir: Optional[str] = field(
        default=None, metadata={"help": "Saving directory of the Tensorflow trainable checkpoints."}
    )
    overwrite_ckpt_dir: bool = field(
        default=False, 
        metadata={
            "help": (
                "Overwrite the content of the ckpt directory."
                "Use this to continue training if ckpt_dir points to a checkpoint directory."
            )
        },
    )
    overwrite_tf_ckpt_dir: bool = field(
        default=False, 
        metadata={
            "help": (
                "Overwrite the content of the TF traninable ckpt directory."
                "Use this to continue training if tf_ckpt_dir points to a checkpoint directory."
            )
        },
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
        """
        -output_dir is used to save the pred files 
        -use overwrite_output_dir to overwrite all precedent results (result files)
        -if ckpt_dir is not defined then use output_dir/ckpt_name to save ckpts
        -set ckpt_dir to separate the pred files and ckpt save dirs 
        """

        if self.ckpt_dir is None and self.ckpt_name is not None:
            self.ckpt_dir = os.path.join(self.output_dir, f'ckpt_{self.ckpt_name}')
        elif self.ckpt_dir is None and self.ckpt_name is None:
            raise ValueError('TrainingArguments: No checkpoint specified !')
        
        if (
            tf.io.gfile.exists(self.ckpt_dir)
            and tf.io.gfile.listdir(self.ckpt_dir)
            and self.do_train
            and not self.overwrite_ckpt_dir
        ):
            raise ValueError(
                f"TrainingArguments: ({self.ckpt_dir}) already exists and is not empty. Use --overwrite_ckpt_dir to overcome."
            )

        if self.overwrite_ckpt_dir:
            tf.io.gfile.rmtree(self.ckpt_dir)
        
        if self.tf_ckpt_dir is None and self.ckpt_name is not None:
            self.tf_ckpt_dir = os.path.join(self.output_dir, f'tf_ckpt_{self.ckpt_name}')
        elif self.tf_ckpt_dir is None and self.ckpt_name is None:
            raise ValueError('TrainingArguments: No TF checkpoint specified !')
    
        if (
            tf.io.gfile.exists(self.tf_ckpt_dir)
            and tf.io.gfile.listdir(self.tf_ckpt_dir)
            and self.do_train
            and not self.overwrite_tf_ckpt_dir
        ):
            raise ValueError(
                f"TrainingArguments: ({self.tf_ckpt_dir}) already exists and is not empty. Use --overwrite_tf_ckpt_dir to overcome."
            )

        if self.overwrite_tf_ckpt_dir:
            tf.io.gfile.rmtree(self.tf_ckpt_dir)