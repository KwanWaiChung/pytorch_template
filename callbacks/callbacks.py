import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict


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
            callback.set_trainer(trainer)

    def on_train_begin(self, logs):
        """

        Args(keys in logs):
            n_epochs (int): Total number of epochs.

        Examples:
            Initialize some objects.

        """
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs):
        """
        Args(keys in logs):
            n_epochs (int): Total number of epochs.

        Example:
            Do ploting.

        """
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, logs):
        """
        Args(keys in logs):
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.

        """
        for callback in self.callbacks:
            callback.on_epoch_begin(logs)

    def on_epoch_end(self, logs):
        """
        Args(keys in logs):
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.

        Returns:
            True to early stop if any callback return True.

        """
        end = False
        for callback in self.callbacks:
            if callback.on_epoch_end(logs):
                end = True
        return end

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
            n_batches (int): Total number of training batches.

        """
        for callback in self.callbacks:
            callback.on_loss_begin(logs)

    def on_loss_end(self, logs):
        """
        Args(keys in logs):
            loss (float): The loss of the current training batch.

        Examples:
            Record the loss of every batch.

        """
        for callback in self.callbacks:
            callback.on_loss_end(logs)

    def on_step_begin(self, logs):
        """
        Args(keys in logs):
            loss (float): The loss of the current training batch.
            last_y_pred (torch.tensor): The last output of the model.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of training batches.

        Examples:
            Gradient clipping.
            Change learning rate.

        """
        for callback in self.callbacks:
            callback.on_step_begin(logs)

    def on_train_batch_begin(self, logs):
        """
        Args(keys in logs):
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of training batches.

        Examples:
            Further custom reshaping the input.

        """
        for callback in self.callbacks:
            callback.on_train_batch_begin(logs)

    def on_train_batch_end(self, logs):
        """
        Args(keys in logs):
            same as on_step_begin
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
            n_batches (int): Total number of testing batches.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.

        """
        for callback in self.callbacks:
            callback.on_train_batch_end(logs)

    def on_test_batch_end(self, logs):
        """
        Args(keys in logs):
            last_y_pred (torch.tensor): The last output of the model.
            val_loss (torch.tensor): The validation loss of current val batch.
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
            n_epochs (int): Total number of epochs.

        Example:
            Do ploting.

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
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.

        Returns:
            True to early stop if any callback return True.

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
            n_batches (int): Total number of training batches.

        """
        pass

    def on_loss_end(self, logs):
        """
        Args(keys in logs):
            loss (float): The loss of the current training batch.

        Examples:
            Record the loss of every batch.

        """
        pass

    def on_step_begin(self, logs):
        """
        Args(keys in logs):
            loss (float): The loss of the current training batch.
            last_y_pred (torch.tensor): The last output of the model.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of training batches.

        Examples:
            Gradient clipping.
            Change learning rate.

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
            n_batches (int): Total number of training batches.

        """
        pass

    def on_train_batch_end(self, logs):
        """
        Args(keys in logs):
            same as on_step_begin
        """
        pass

    def on_test_batch_begin(self, logs):
        """
        Args(keys in logs):
            last_X (torch.tensor): The last input of the model.
            last_y_true (torch.tensor): The last target of the model.
            batch (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of testing batches.
            n_epochs (int): Total number of epochs.
            epoch (int): Index of the current epoch, which starts at 1.

        """
        pass

    def on_test_batch_end(self, logs):
        """
        Args(keys in logs):
            last_y_pred (torch.tensor): The last output of the model.
            val_loss (torch.tensor): The validation loss of current val batch.
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
    """ Records events and metrics.
    This callback is automatically applied to every trainer.
    """

    def __init__(self):
        super().__init__()
        self.history = defaultdict(list)
        self.epoch = []

    def on_epoch_end(self, logs=None):
        # training metrics
        log_data = {
            metric.name: logs[metric.name] for metric in self.trainer.metrics
        }
        log_data["loss"] = logs["loss"]

        # validation metrics
        log_data.update(
            {k: v for k, v in logs.items() if k.startswith("val_")}
        )

        #  if "val_loss" in logs:
        #      log_data["val_loss"] = logs["val_loss"]

        self.epoch.append(logs["epoch"])
        for k, v in log_data.items():
            self.history[k].append(v)
        return False


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
            metric.name: f"{logs[metric.name]}:.4f"
            for metric in self.trainer.metrics
        }
        log_data["loss"] = logs["loss"]

        # validation metrics
        log_data.update(
            {k: "%.4f" % v for k, v in logs.items() if k.startswith("val_")}
        )

        if "val_loss" in logs:
            log_data["val_loss"] = logs["val_loss"]

        self.progbar.set_postfix(log_data)
        self.progbar.update(0)
        self.progbar.close()
        return False

    #  def on_train_batch_begin(self, logs=None):
    #      # update and increment progbar
    #      self.progbar.update(1)

    def on_train_batch_end(self, logs=None):
        """Update the training metrics"""
        self.progbar.update(1)
        log_data = {
            metric.name: f"{logs[metric.name]}:.4f"
            for metric in self.trainer.metrics
        }
        log_data["loss"] = logs["loss"]
        self.progbar.set_postfix(log_data)
