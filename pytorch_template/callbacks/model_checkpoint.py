from . import Callback, EarlyStopping
from typing import Dict
from ..utils import getlogger
import os
import re
import torch


class ModelCheckpoint(Callback):
    """A Callback that save model weights during training.

        This callback is used in training to save model or weights.
        It can resume training at the beginning(on_train_begin).

        A few options this callback provides include:

        - Whether to only keep the model that has achieved the
            "best performance" so far, or whether to save the model at the
            end of every epoch regardless of performance.
        - Definition of 'best'; which quantity to monitor and whether it should
            be maximized or minimized.
        - Whether to restore weights at the beginning of training.

    """

    def __init__(
        self,
        filepath="save/checkpoint_{epoch:02d}_{val_loss:.2f}.pth",
        monitor: str = "val_loss",
        mode="min",
        save_best_only: bool = False,
        save_weights_only: bool = True,
        load_weights_on_restart: bool = False,
        max_save: int = -1,
        save_dict={},
    ):
        """
        Args:
            filepath (str): The path to save the model file. It can contain
                named formatting options, which will be filled with value in
                logs(passed in on_epoch_end).
                Default: save/checkpont_{epoch:02d}_{val_loss:.2f}.pth
            monitor (str): The metric to monitor on. Must be in trainer.logs.
            mode (str): Either `min` or `max`. If save_best_only=True, the
                decision to overwrite the current save file is made based on
                either the maximization or the minimization of the monitored
                quantity.  For val_acc, this should be max, for val_loss this
                should be min, etc.
            save_best_only (bool): If True, it will only save when monitored
                value has improved. Otherwise, it will save every epoch.
            save_weights_only (bool): If True, it will save only the weights.
                Otherwise, the whole model will be saved.
            load_weights_on_restart(bool): If True, it will resume training
                from the last checkpoint.
            max_save (int): The max number of models to save. Older model
                checkpoints will be overwritten if necessary. If the value
                is -1, then it will have no limit.
            save_dict (dict): Additional object that need to be save.

        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode.lower()
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.load_weights_on_restart = load_weights_on_restart
        self.max_save = max_save
        self.save_dict = save_dict
        self.logger = getlogger(__name__)
        self.best_score = float("inf") if self.mode == "min" else float("-inf")

        if self.max_save > 0:
            self.old_files = []

        super().__init__()

    def _get_filepath(self, logs):
        try:
            filepath = self.filepath.format(**logs)
        except KeyError as e:
            raise KeyError(
                "Failed to format the filepath: `{}`. "
                "Reason: `{}` is not provided during training".format(
                    self.filepath, e
                )
            )
        return filepath

    def _get_most_recent_checkpoint(self):
        """Returns the most recent checkpoint matching pattern.

        Implementation referenced from tensorflow
        _get_most_recently_modified_file_matching_pattern()
        """
        dirname = os.path.dirname(self.filepath)
        basename = os.path.basename(self.filepath)
        basename_regex = "^" + re.sub(r"{.*}", r".*", basename) + "$"
        latest_modtime = 0
        latest_filename = None
        for filename in os.listdir(dirname):
            if not re.match(basename_regex, filename):
                continue
            filepath = os.path.join(dirname, filename)
            modtime = os.path.getmtime(filepath)
            if modtime > latest_modtime:
                latest_modtime = modtime
                latest_filename = filepath
        return latest_filename

    def _save(self, obj, logs):
        """Save model to checkpoint_{epoch}_{metric}.pth in self.directory.


        Args:
            epoch (int): Index of the current epoch, which starts at 1.
            metric (float): The metric score of current epoch.
            obj: Either a dict containing all the necessary state_dicts if
                `save_weights_only` or the model object otherwise.

        """
        filepath = self._get_filepath(logs)
        torch.save(obj, filepath)

    def on_train_begin(self, logs):
        if not self.load_weights_on_restart:
            return

        filepath = self._get_most_recent_checkpoint()
        if not filepath:
            self.logger.info(
                "%s is not found. Start training from scratch.", self.filepath
            )
            return
        checkpoint = torch.load(filepath)
        self.trainer.model.load_state_dict(checkpoint["model_state_dict"])
        self.trainer.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        self.best_score = checkpoint[self.monitor]
        # restore early stopping
        for value in checkpoint.values():
            if isinstance(value, EarlyStopping):
                for i, c in enumerate(self.trainer.callbacks.callbacks):
                    if isinstance(c, EarlyStopping):
                        self.trainer.callbacks.callbacks[i] = value
                        value.set_trainer(self.trainer)
                        self.logger.debug("Restored EarlyStopping callback")

        # checkpoint finished training on checkpoint['epoch']
        logs["epoch"] = checkpoint["epoch"] + 1
        self.logger.info(
            "Model weights loaded. Resuming training at %d epoch",
            logs["epoch"],
        )

    def on_epoch_end(self, logs):
        save_dict = {"epoch": logs["epoch"]}
        save_dict.update(self.save_dict)
        improved = False
        if self.monitor not in logs:
            raise ValueError(
                f"Metric `{self.monitor}` is not provided during training."
            )
        score = logs[self.monitor]
        if (score < self.best_score and self.mode == "min") or (
            score > self.best_score and self.mode == "max"
        ):
            improved = True
            self.best_score = score
        save_dict[self.monitor] = self.best_score

        if self.save_weights_only:
            save_dict.update(self.trainer._getstate())
        else:
            save_dict["model"] = self.trainer.model
        if not self.save_best_only or (self.save_best_only and improved):
            self._save(save_dict, logs)
        return False
