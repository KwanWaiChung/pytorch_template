from . import Callback
from ..utils import getlogger


class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving."""

    def __init__(
        self, monitor="val_loss", mode="min", min_delta=0, patience=5
    ):
        """
        Args:
            monitor (str): The metric to monitor on.. Must be in trainer.logs.
            mode (str): Either `min` or `max`. If save_best_only=True, the
                decision to overwrite the current save file is made based on
                either the maximization or the minimization of the monitored
                quantity.  For val_acc, this should be max, for val_loss this
                should be min, etc.
            min_delta (float) : Minimum change in monitored value to qualify
                as improvement, i.e. an absolute change of less than min_delta,
                will count as no improvement. Must be a positive number.
            patience (int): Number of epochs with no improvement after which
                training will be stopped.
        """
        self.monitor = monitor
        self.mode = mode.lower()
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_score = float("inf") if self.mode == "min" else float("-inf")
        self.logger = getlogger(__name__)
        super().__init__()

    def on_train_begin(self, logs):
        self.wait = 0
        self.best_score = float("inf") if self.mode == "min" else float("-inf")

    def on_epoch_end(self, logs):
        if self.monitor not in logs:
            raise ValueError(
                f"Metric `{self.monitor}` is not provided during training."
            )
        score = logs[self.monitor]
        if (
            (self.best_score - score) > self.min_delta and self.mode == "min"
        ) or (
            (score - self.best_score) > self.min_delta and self.mode == "max"
        ):
            self.logger.debug(
                "%s improved on epoch %d", self.monitor, logs["epoch_idx"]
            )
            self.best_score = score
            self.wait = 0
        elif self.wait >= self.patience:
            return True
        else:
            self.wait += 1
            self.logger.debug(
                "No improved on epoch %d, wait=%d",
                logs["epoch_idx"],
                self.wait,
            )
        return False

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "trainer"}

    def on_save_checkpoint(self, logs):
        logs["save_dict"]["early_stopping"] = self.state_dict()

    def on_load_checkpoint(self, logs):
        self.__dict__.update(logs["save_dict"]["early_stopping"])
