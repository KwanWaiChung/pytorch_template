class CallbackHandler:
    """Container that "abstracting a list of callbacks

    Attributes:
        callbacks (List[Callback]): The List of callbacks to call in order.
        trainer (Trainer): The trainer object for accessing some necessary
            attributes for some method.
    """

    def __init__(self, callbacks, trainer):
        self.callbacks = callbacks
        self.trainer = trainer
        for callback in callbacks:
            callback.set_trainer(trainer)

    def on_train_begin(self, logs):
        """Called before everything.

        Args(keys in logs):
            n_epochs (int): Total number of epochs.

        Examples:
            Initialize some variables and constants.

        """
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_epoch_begin(self, logs):
        """Called at the beginning of each epoch.

        Args(keys in logs):
            n_batches (int): Total number of training batches.
            epoch_idx (int): Index of the current epoch, which starts at 1.
            n_epochs (int): Total number of epochs.

        """
        for callback in self.callbacks:
            callback.on_epoch_begin(logs)

    def on_train_batch_begin(self, logs):
        """Called before the output and loss are computed.

        Args(keys in logs):
            batch_dict (Dict[str, torch.Tensor]): The last input of the model.
            batch_idx (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of training batches.
            epoch_idx (int): Index of the current epoch, which starts at 1.
            n_epochs (int): Total number of epochs.

        Examples:
            Moving tensors to gpu.

        """
        for callback in self.callbacks:
            callback.on_train_batch_begin(logs)

    def on_loss_end(self, logs):
        """Called after the output and loss are computed but before backprop.

        Args(keys in logs):
            **other: All the outputs provided in training_step.
            loss (float): The avg loss of current training batch.
            y_pred (torch.tensor): The last output of the model.
            y_true (torch.tensor): The last target of the model.
            batch_size (int): The batch size of current batch.
            batch_dict (torch.tensor): The last input of the model.
            batch_idx (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of training batches.
            epoch_idx (int): Index of the current epoch, which starts at 1.
            n_epochs (int): Total number of epochs.

        """
        for callback in self.callbacks:
            callback.on_loss_end(logs)

    def on_step_begin(self, logs):
        """Called after backprop but before the optimizer step.

        Args(keys in logs):
            **other: All the outputs provided in training_step.
            loss (float): The avg loss of current training batch.
            y_pred (torch.tensor): The last output of the model.
            y_true (torch.tensor): The last target of the model.
            batch_dict (torch.tensor): The last input of the model.
            batch_idx (int): Index of the batch in current epoch,
                which starts at 1.
            batch_size (int): The batch size of current batch.
            n_batches (int): Total number of training batches.
            epoch_idx (int): Index of the current epoch, which starts at 1.
            n_epochs (int): Total number of epochs.

        Examples:
            Gradient clipping.
            Change learning rate.

        """
        for callback in self.callbacks:
            callback.on_step_begin(logs)

    def on_train_batch_end(self, logs):
        """Called after optimizer step and at the end of training this batch.

        Args(keys in logs):
            `metric name` (float): Those metrics that passed to trainer.
            **other: All the outputs provided in training_step.
            loss (float): The avg loss of current training batch.
            y_pred (torch.tensor): The last output of the model.
            y_true (torch.tensor): The last target of the model.
            batch_dict (torch.tensor): The last input of the model.
            batch_idx (int): Index of the batch in current epoch,
                which starts at 1.
            batch_size (int): The batch size of current batch.
            n_batches (int): Total number of training batches.
            epoch_idx (int): Index of the current epoch, which starts at 1.
            n_epochs (int): Total number of epochs.

        Examples:
            Keep track of the loss.

        """
        for callback in self.callbacks:
            callback.on_train_batch_end(logs)

    def on_val_begin(self, logs):
        """Called before validating in each epoch.

        Args(keys in logs):
            epoch_idx (int): Index of the current epoch, which starts at 1.
            n_epochs (int): Total number of epochs.
            n_batches (int): Total number of validation batches

        """
        for callback in self.callbacks:
            callback.on_val_begin(logs)

    def on_val_batch_begin(self, logs):
        """Called before the val output and loss are computed.

        Args(keys in logs):
            batch_dict (torch.tensor): The last input of the model.
            batch_idx (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of validation batches.
            n_epochs (int): Total number of epochs.
            batch_size (int): The batch size of current batch.
            epoch_idx (int): Index of the current epoch, which starts at 1.

        """
        for callback in self.callbacks:
            callback.on_val_batch_begin(logs)

    def on_val_batch_end(self, logs):
        """Called after the val loss and metrics computed, at the end
            of this val batch.

        Args(keys in logs):
            val_`metric name` (float): Those metrics that passed to trainer.
            val_loss (float): The avg validation loss of
                current val batch.
            y_pred (torch.tensor): The last output of the model.
            y_true (torch.tensor): The last target of the model.
            batch_dict (torch.tensor): The last input of the model.
            batch_idx (int): Index of the batch in current epoch,
                which starts at 1.
            batch_size (int): The batch size of current batch.
            n_batches (int): Total number of testing batches.
            n_epochs (int): Total number of epochs.
            batch_size (int): The batch size of current batch.
            epoch_idx (int): Index of the current epoch, which starts at 1.

        Examples:
            Keep track of the loss.

        """
        for callback in self.callbacks:
            callback.on_val_batch_end(logs)

    def on_val_end(self, logs):
        """Called after validation is completed .

        Args(keys in logs):
            val_`metric name` (float): Those metrics that passed to trainer.
            val_loss (float): The avg validation loss of
                current val batch.
            y_pred (torch.tensor): The last output of the model.
            y_true (torch.tensor): The last target of the model.
            batch_dict (torch.tensor): The last input of the model.
            batch_idx (int): Index of the batch in current epoch,
                which starts at 1.
            batch_size (int): The batch size of current batch.
            n_batches (int): Total number of testing batches.
            n_epochs (int): Total number of epochs.
            batch_size (int): The batch size of current batch.
            epoch_idx (int): Index of the current epoch, which starts at 1.

        """
        for callback in self.callbacks:
            callback.on_val_end(logs)

    def on_epoch_end(self, logs):
        """Called at the end of an epoch.

        Args(keys in logs):
            val_`metric name` (float): The validation metrics for this epoch.
            val_loss (float): The avg validation loss of this epoch.
            `metric name` (float): The training metrics of this epoch.
            loss (float): The avg training loss for this epoch.
            n_epochs (int): Total number of epochs.
            epoch_idx (int): Index of the current epoch, which starts at 1.

        Returns:
            True to early stop if any callback return True.

        """
        end = False
        for callback in self.callbacks:
            if callback.on_epoch_end(logs):
                end = True
        return end

    def on_train_end(self, logs):
        """Called at the end of training.

        Args(keys in logs):
            n_epochs (int): Total number of epochs.
            epoch_idx (int): Index of the current epoch, which starts at 1.

        Examples:
            Saving model and stuff.

        """
        for callback in self.callbacks:
            callback.on_train_end(logs)


class Callback:
    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, logs):
        """Called before everything.

        Args(keys in logs):
            n_epochs (int): Total number of epochs.

        Examples:
            Initialize some variables and constants.

        """
        pass

    def on_epoch_begin(self, logs):
        """Called at the beginning of each epoch.

        Args(keys in logs):
            n_batches (int): Total number of training batches.
            epoch_idx (int): Index of the current epoch, which starts at 1.
            n_epochs (int): Total number of epochs.

        """
        pass

    def on_train_batch_begin(self, logs):
        """Called before the output and loss are computed.

        Args(keys in logs):
            batch_dict (torch.tensor): The last input of the model.
            batch_idx (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of training batches.
            epoch_idx (int): Index of the current epoch, which starts at 1.
            n_epochs (int): Total number of epochs.

        Examples:
            Moving tensors to gpu.

        """
        pass

    def on_loss_end(self, logs):
        """Called after the output and loss are computed but before backprop.

        Args(keys in logs):
            **other: All the outputs provided in training_step.
            loss (float): The avg loss of current training batch.
            y_pred (torch.tensor): The last output of the model.
            y_true (torch.tensor): The last target of the model.
            batch_size (int): The batch size of current batch.
            batch_dict (torch.tensor): The last input of the model.
            batch_idx (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of training batches.
            epoch_idx (int): Index of the current epoch, which starts at 1.
            n_epochs (int): Total number of epochs.

        """
        pass

    def on_loss_begin(self, logs):
        """Called after the output and loss are computed but before backprop.

        Args(keys in logs):
            **other: All the outputs provided in training_step.
            loss (float): The avg loss of current training batch.
            y_pred (torch.tensor): The last output of the model.
            y_true (torch.tensor): The last target of the model.
            batch_dict (torch.tensor): The last input of the model.
            batch_idx (int): Index of the batch in current epoch,
                which starts at 1.
            batch_size (int): The batch size of current batch.
            n_batches (int): Total number of training batches.
            epoch_idx (int): Index of the current epoch, which starts at 1.
            n_epochs (int): Total number of epochs.

        """
        pass

    def on_step_begin(self, logs):
        """Called after backprop but before the optimizer step.

        Args(keys in logs):
            **other: All the outputs provided in training_step.
            loss (float): The avg loss of current training batch.
            y_pred (torch.tensor): The last output of the model.
            y_true (torch.tensor): The last target of the model.
            batch_dict (torch.tensor): The last input of the model.
            batch_idx (int): Index of the batch in current epoch,
                which starts at 1.
            batch_size (int): The batch size of current batch.
            n_batches (int): Total number of training batches.
            epoch_idx (int): Index of the current epoch, which starts at 1.
            n_epochs (int): Total number of epochs.

        Examples:
            Gradient clipping.
            Change learning rate.

        """
        pass

    def on_train_batch_end(self, logs):
        """Called after optimizer step and at the end of training this batch.

        Args(keys in logs):
            `metric name` (float): Those metrics that passed to trainer.
            **other: All the outputs provided in training_step.
            loss (float): The avg loss of current training batch.
            y_pred (torch.tensor): The last output of the model.
            y_true (torch.tensor): The last target of the model.
            batch_dict (torch.tensor): The last input of the model.
            batch_idx (int): Index of the batch in current epoch,
                which starts at 1.
            batch_size (int): The batch size of current batch.
            n_batches (int): Total number of training batches.
            epoch_idx (int): Index of the current epoch, which starts at 1.
            n_epochs (int): Total number of epochs.

        Examples:
            Keep track of the loss.

        """
        pass

    def on_val_begin(self, logs):
        """Called before validating in each epoch.

        Args(keys in logs):
            epoch_idx (int): Index of the current epoch, which starts at 1.
            n_epochs (int): Total number of epochs.
            n_batches (int): Total number of validation batches

        """
        pass

    def on_val_batch_begin(self, logs):
        """Called before the val output and loss are computed.

        Args(keys in logs):
            batch_dict (torch.tensor): The last input of the model.
            batch_idx (int): Index of the batch in current epoch,
                which starts at 1.
            n_batches (int): Total number of validation batches.
            n_epochs (int): Total number of epochs.
            batch_size (int): The batch size of current batch.
            epoch_idx (int): Index of the current epoch, which starts at 1.

        """
        pass

    def on_val_batch_end(self, logs):
        """Called after the val loss and metrics computed, at the end
            of this val batch.

        Args(keys in logs):
            val_`metric name` (float): Those metrics that passed to trainer.
            val_loss (float): The avg validation loss of
                current val batch.
            y_pred (torch.tensor): The last output of the model.
            y_true (torch.tensor): The last target of the model.
            batch_dict (torch.tensor): The last input of the model.
            batch_idx (int): Index of the batch in current epoch,
                which starts at 1.
            batch_size (int): The batch size of current batch.
            n_batches (int): Total number of testing batches.
            n_epochs (int): Total number of epochs.
            batch_size (int): The batch size of current batch.
            epoch_idx (int): Index of the current epoch, which starts at 1.

        Examples:
            Keep track of the loss.

        """
        pass

    def on_val_end(self, logs):
        """Called after validation is completed .

        Args(keys in logs):
            val_`metric name` (float): Those metrics that passed to trainer.
            val_loss (float): The avg validation loss of
                current val batch.
            y_pred (torch.tensor): The last output of the model.
            y_true (torch.tensor): The last target of the model.
            batch_dict (torch.tensor): The last input of the model.
            batch_idx (int): Index of the batch in current epoch,
                which starts at 1.
            batch_size (int): The batch size of current batch.
            n_batches (int): Total number of testing batches.
            n_epochs (int): Total number of epochs.
            batch_size (int): The batch size of current batch.
            epoch_idx (int): Index of the current epoch, which starts at 1.

        """
        pass

    def on_epoch_end(self, logs):
        """Called at the end of an epoch.

        Args(keys in logs):
            val_`metric name` (float): The validation metrics for this epoch.
            val_loss (float): The avg validation loss of this epoch.
            `metric name` (float): The training metrics of this epoch.
            loss (float): The avg training loss for this epoch.
            n_epochs (int): Total number of epochs.
            epoch_idx (int): Index of the current epoch, which starts at 1.

        Returns:
            True to early stop if any callback return True.

        """
        return False

    def on_train_end(self, logs):
        """Called at the end of training.

        Args(keys in logs):
            n_epochs (int): Total number of epochs.
            epoch_idx (int): Index of the current epoch, which starts at 1.

        Examples:
            Saving model and stuff.

        """
        pass
