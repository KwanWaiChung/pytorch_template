from typing import List, Union, Callable, Dict
from ..callbacks import CallbackHandler, History, ProgressBar
import torch
import torch.nn as nn


class BaseTrainer:
    """Base class for all trainers.

    It is an abstraction that combines all the modules that training
    needs, (model, optimizer, criterion, metrics)

    Attributes:
        model (nn.Module): The pytorch model.
        criterion (nn.Module): The loss function.
        optimizer (optim.optimizer): The optimizer to use.
        device (torch.device): The device(gpu/cpu) to use.
        metrics (Dict[str, Callable]): A Dict that maps metrics name to
            its callable function, which accepts (y_true, y_pred) and returns
            a float.
        callbacks (CallbackHandler): An object that will handles the callback.

    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: "optimizer",
        device: torch.device = None,
        metrics: List["Metric"] = [],
        callbacks: List["Callback"] = [],
    ):
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.history = History()
        self.callbacks = CallbackHandler(
            metrics + callbacks + [self.history, ProgressBar()], self
        )
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.use_val = False
        self.logs = {}

    def fit(
        self, train_dl: "Iterator", n_epochs: int, val_dl: "Iterator" = None
    ):
        if val_dl is not None:
            self.use_val = True
        self.logs["n_epochs"] = n_epochs
        self.callbacks.on_train_begin(self.logs)
        for i in range(self.logs.get("epoch", 1), n_epochs + 1):
            self.logs["epoch"] = i
            if self.fit_epoch(train_dl, val_dl):
                break
            if self.callbacks.on_epoch_end(self.logs):
                break
        self.callbacks.on_train_end(self.logs)

    def fit_epoch(self, train_dl: "Iterator", val_dl: "Iterator"):
        """Run one training epoch.

        Returns:
            bool: True, if early stop after trained certain batches.
        """
        self.logs["n_batches"] = len(train_dl)
        self.callbacks.on_epoch_begin(self.logs)
        if self._train(train_dl):
            return True
        if self.use_val:
            self._validate(val_dl)
        return False

    def _train(self, train_dl):
        """Training for one epoch.

        Returns:
            bool: True, if early stop after trained certain batches.
        """
        self.model.train()
        for i, (train_X, train_y) in enumerate(train_dl, 1):
            train_X = train_X.to(self.device)
            train_y = train_y.to(self.device)
            self.logs["last_X"] = train_X
            self.logs["last_y_true"] = train_y
            self.logs["batch"] = i
            self.logs["batch_size"] = train_X.shape[0]
            self.callbacks.on_train_batch_begin(self.logs)

            output = self.model(self.logs["last_X"])
            self.logs["last_y_pred"] = output

            self.callbacks.on_loss_begin(self.logs)
            loss = self.criterion(
                self.logs["last_y_pred"], self.logs["last_y_true"]
            )
            # loss.data is the avg loss of this batch
            self.logs["loss"] = loss.item()
            self.callbacks.on_loss_end(self.logs)

            self.optimizer.zero_grad()
            loss.backward()
            self.callbacks.on_step_begin(self.logs)
            self.optimizer.step()
            if self.callbacks.on_train_batch_end(self.logs):
                return True
        return False

    def _validate(self, val_dl):
        self.model.eval()
        self.logs["n_batches"] = len(val_dl)
        self.callbacks.on_val_begin(self.logs)
        with torch.no_grad():
            for i, (val_X, val_y) in enumerate(val_dl, 1):
                val_X = val_X.to(self.device)
                val_y = val_y.to(self.device)
                self.logs["last_X"] = val_X
                self.logs["last_y_true"] = val_y
                self.logs["batch"] = i
                self.logs["batch_size"] = val_X.shape[0]
                self.callbacks.on_test_batch_begin(self.logs)

                output = self.model(self.logs["last_X"])

                self.logs["last_y_pred"] = output
                self.callbacks.on_loss_begin(self.logs)
                loss = self.criterion(
                    self.logs["last_y_pred"], self.logs["last_y_true"]
                )
                self.callbacks.on_loss_end(self.logs)
                self.logs["val_loss"] = loss.item()
                self.callbacks.on_test_batch_end(self.logs)

    def _getstate(self):
        """Get all the related state dicts for saving

        Returns:
            dict
        """
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
