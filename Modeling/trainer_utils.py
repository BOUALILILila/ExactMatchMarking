import  os
import numpy as np
import tensorflow as tf
import random

from absl import logging

logging.set_verbosity(logging.INFO)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, best_score= None, 
                verbose=False, delta=0, save_dir = None, 
                ckpt_manager= None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            best_score (float): Continue training from a ckpt that achieved best_score.
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_dir (str): Directory to save checkpoints.
            ckpt_manager (tf.train.CheckpointManager): tensforflow ckpt manager for saving ckpts.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_dir = save_dir
        self.ckpt_manager = ckpt_manager
        self.early_stop = False

    def __call__(self, score, model, global_step):
        if self.best_score is None :
            self.best_score = score
            self.save_checkpoint(global_step, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                #stop
                self.early_stop = True          
        else:
            self.best_score = score
            self.save_checkpoint(global_step, model)
            self.counter = 0

    def save_checkpoint(self, step, model):
        '''Saves model when validation loss decreases.'''
        # 1: save tf trainable ckpt 
        ckpt_save_path = self.ckpt_manager.save()
        logging.info(f">> Saving checkpoint for step {step} at {ckpt_save_path}")

        # 2: save transformer ckpt
        save_chkpt_dir = os.path.join(self.save_dir, f"checkpoint_{self.best_score}-{step}")
        if not os.path.exists(save_chkpt_dir):
            os.makedirs(save_chkpt_dir)
        model.save_pretrained(save_chkpt_dir)
        if self.verbose:
            logging.info(f">> Saving model checkpoint to {save_chkpt_dir}")

