import torch.nn as nn
from tqdm import tqdm


class CallbackHandler:
    """Container that "abstracting a list of callbacks

    Attributes:
        callbacks ():
        trainer ():
    """

    def __init__(self, callbacks, trainer):
        self.callbacks = callbacks
        self.trainer = trainer
        for callback in callbacks:
            callback.set_model(trainer)

    def on_train_begin(self, logs):
        """

        Args(keys in logs):
            n_epochs (int): Total number of epochs.

        Examples:
            Initialize some objects.
        """
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, **logs):
        """
        Args(keys in logs):
            last_loss (float): The loss of the current batch.
            last_y_pred (torch.tensor): The last output of the model.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.

        Example:
            Record the metrics
        """
        for callback in self.callbacks:
            callback.on_train_end(**logs)

    def on_epoch_begin(self, logs):
        """
        Args(keys in logs):
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            n_batches (int): Total number of batches.
        """
        for callback in self.callbacks:
            callback.on_epoch_begin(logs)

    def on_epoch_end(self, logs):
        """
        Args(keys in logs):
            last_loss (float): The loss of the current batch.
            last_y_pred (torch.tensor): The last output of the model.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.

        Example:
            Record the metrics
        """
        res = False
        for callback in self.callbacks:
            if callback.on_epoch_end(logs):
                res = True
        return res

    def on_loss_begin(self, logs):
        """
        Args(keys in logs):
            last_y_pred (torch.tensor): The last output of the model.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.
        """
        for callback in self.callbacks:
            callback.on_loss_begin(logs)

    def on_loss_end(self, logs):
        """
        Args(keys in logs):
            last_loss (float): The loss of the current batch.
            last_y_pred (torch.tensor): The last output of the model.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.

        Examples:
            Record the loss of every batch.
        """
        for callback in self.callbacks:
            callback.on_loss_end(logs)

    def on_step_begin(self, logs):
        """
        Args(keys in logs):
            last_loss (float): The loss of the current batch.
            last_y_pred (torch.tensor): The last output of the model.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.

        Examples:
            Gradient clipping
            Change learning rate
        """
        for callback in self.callbacks:
            callback.on_step_begin(logs)

    def on_step_end(self, logs):
        """
        Args(keys in logs):
            last_loss (float): The loss of the current batch.
            last_y_pred (torch.tensor): The last output of the model.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.
        """
        for callback in self.callbacks:
            callback.on_step_end(logs)

    def on_train_batch_begin(self, logs):
        """
        Args(keys in logs):
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.
        """
        for callback in self.callbacks:
            callback.on_train_batch_begin(logs)

    def on_train_batch_end(self, logs):
        """
        Args(keys in logs):
            last_loss (float): The loss of the current batch.
            last_y_pred (torch.tensor): The last output of the model.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.

        Example:
            Record the metrics
        """
        for callback in self.callbacks:
            callback.on_train_batch_end(logs)

    def on_test_batch_begin(self, logs):
        """
        Args(keys in logs):
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.
        """
        for callback in self.callbacks:
            callback.on_train_batch_end(logs)

    def on_test_batch_end(self, logs):
        """
        Args(keys in logs):
            last_y_pred (torch.tensor): The last output of the model.
            last_loss (torch.tensor): The last loss of the model.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.
        """
        for callback in self.callbacks:
            callback.on_test_batch_end(logs)


class Callback:
    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, logs):
        """No particular usecase for now.

        Args(keys in logs):
            n_epochs (int): Total number of epochs.
        """
        pass

    def on_train_end(self, logs):
        """
        Args(keys in logs):
            last_loss (float): The loss of the current batch.
            last_y_pred (torch.tensor): The last output of the model.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.

        Example:
            Record the metrics
        """
        pass

    def on_epoch_begin(self, logs):
        """
        Args(keys in logs):
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
        """
        pass

    def on_epoch_end(self, logs):
        """
        Args(keys in logs):
            last_loss (float): The loss of the current batch.
            last_y_pred (torch.tensor): The last output of the model.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.

        Example:
            Record the metrics
        """
        return False

    def on_loss_begin(self, logs):
        """
        Args(keys in logs):
            last_y_pred (torch.tensor): The last output of the model.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.
        """
        pass

    def on_loss_end(self, logs):
        """
        Args(keys in logs):
            last_loss (float): The loss of the current batch.
            last_y_pred (torch.tensor): The last output of the model.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.

        Examples:
            Record the loss of every batch.
        """
        pass

    def on_step_begin(self, logs):
        """
        Args(keys in logs):
            last_loss (float): The loss of the current batch.
            last_y_pred (torch.tensor): The last output of the model.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.

        Examples:
            Gradient clipping
            Change learning rate
        """
        pass

    def on_step_end(self, logs):
        """
        Args(keys in logs):
            last_loss (float): The loss of the current batch.
            last_y_pred (torch.tensor): The last output of the model.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.
        """
        pass

    def on_train_batch_begin(self, logs):
        """
        Args(keys in logs):
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.
        """
        pass

    def on_train_batch_end(self, logs):
        """
        Args(keys in logs):
            last_loss (float): The loss of the current batch.
            last_y_pred (torch.tensor): The last output of the model.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.

        Example:
            Record the metrics
        """
        pass

    def on_test_batch_begin(self, logs):
        """
        Args(keys in logs):
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.
        """
        pass

    def on_test_batch_end(self, logs):
        """
        Args(keys in logs):
            last_y_pred (torch.tensor): The last output of the model.
            last_loss (torch.tensor): The last loss of the model.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of batches.
        """
        pass


class ParallelTrainer(Callback):
    def on_train_begin(self, logs):
        self.trainer.model = nn.DataParallel(self.trainer.model).to(
            self.trainer.device
        )

    def on_train_end(self, logs):
        self.trainer.model = self.trainer.model.module


class History(Callback):
    pass


class ProgressBar(Callback):
    def __init__(self):
        """
        TQDM Progress Bar callback
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
            metric.name: f"{logs[metric.name]}:.4f"
            for metric in self.trainer.metrics
        }
        log_data["loss"] = logs["last_loss"]

        # validation metrics
        log_data.update(
            {k: "%.4f" % v for k, v in logs.items() if k.startswith("val_")}
        )

        if "val_loss" in logs:
            log_data["val_loss"] = logs["val_loss"]

        self.progbar.set_postfix(log_data)
        self.progbar.update()
        self.progbar.close()
        return False

    def on_train_batch_begin(self, logs=None):
        # update and increment progbar
        self.progbar.update(1)

    def on_train_batch_end(self, logs=None):
        """Update the training metrics"""
        log_data = {
            metric.name: f"{logs[metric.name]}:.4f"
            for metric in self.trainer.metrics
        }
        log_data["loss"] = logs["last_loss"]
        self.progbar.set_postfix(log_data)
