from . import Callback
from tqdm import tqdm


class ProgressBar(Callback):
    def __init__(self):
        """ TQDM Progress Bar callback

        This callback is automatically applied to every trainer
        """
        self.progbar = None
        super().__init__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progbar:
            self.progbar.close()

    def on_epoch_begin(self, logs):
        """Initialize the progress bar"""
        self.progbar = tqdm(total=logs["n_batches"], unit=" batches")
        self.progbar.set_description(
            "Epoch %i/%i" % (logs["epoch"], logs["n_epochs"])
        )

    def on_epoch_end(self, logs=None):
        # training metrics
        log_data = {
            metric.name: f"{logs[metric.name]:.4f}"
            for metric in self.trainer.metrics
        }
        log_data["loss"] = self.trainer.history.history["loss"][-1]

        # validation metrics
        if self.trainer.use_val:
            log_data.update(
                {
                    k: "%.4f" % v
                    for k, v in logs.items()
                    if k.startswith("val_")
                }
            )
            if "val_loss" in logs:
                log_data["val_loss"] = self.trainer.history.history[
                    "val_loss"
                ][-1]

        self.progbar.set_postfix(log_data)
        self.progbar.update(0)
        self.progbar.close()
        return False

    def on_train_batch_end(self, logs=None):
        """Update the training metrics"""
        log_data = {
            metric.name: f"{logs[metric.name]:.4f}"
            for metric in self.trainer.metrics
        }
        log_data["loss"] = self.trainer.history.loss
        self.progbar.set_postfix(log_data)
        self.progbar.update(1)
        return False
