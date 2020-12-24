from . import Callback
from collections import defaultdict


class History(Callback):
    """Records events and metrics.

    This callback is automatically applied to every trainer.
    It will convert the batch loss in `trainer.logs` to average
    loss across all batches.

    Attributes:
        history['loss'] (List): The avg training loss of seen examples.
        history['val_loss'] (List): The avg val loss of seen examples.

    """

    def __init__(self):
        super().__init__()
        self.history = defaultdict(list)
        self.epoch = []

    def on_epoch_begin(self, logs=None):
        self.train_samples = 0
        # total training loss of seen examples
        self.loss = 0

    def on_loss_end(self, logs):
        logs["loss"] = logs["loss"].item()

    def on_train_batch_end(self, logs=None):
        self.loss = (
            logs["loss"] * logs["batch_size"] + self.loss * self.train_samples
        ) / (logs["batch_size"] + self.train_samples)
        self.train_samples += logs["batch_size"]
        logs["loss"] = self.loss

    def on_val_begin(self, logs=None):
        self.val_samples = 0
        self.val_loss = 0

    def on_val_batch_end(self, logs=None):
        self.val_loss = (
            logs["val_loss"] * logs["batch_size"]
            + self.val_loss * self.val_samples
        ) / (self.val_samples + logs["batch_size"])
        self.val_samples += logs["batch_size"]
        logs["val_loss"] = self.val_loss

    def on_test_begin(self, logs=None):
        self.test_samples = 0
        self.test_loss = 0

    def on_test_batch_end(self, logs=None):
        self.test_loss = (
            logs["test_loss"] * logs["batch_size"]
            + self.test_loss * self.test_samples
        ) / (self.test_samples + logs["batch_size"])
        self.test_samples += logs["batch_size"]
        logs["test_loss"] = self.test_loss

    def on_epoch_end(self, logs=None):
        # training metrics
        log_data = {
            metric.name: logs[metric.name] for metric in self.trainer.metrics
        }
        log_data["loss"] = self.loss

        # validation metrics
        if self.trainer.use_val:
            log_data.update(
                {k: v for k, v in logs.items() if k.startswith("val_")}
            )
            log_data["val_loss"] = self.val_loss

        self.epoch.append(logs["epoch_idx"])
        for k, v in log_data.items():
            self.history[k].append(v)
        return False
