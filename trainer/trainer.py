from typing import List, Union, Callable, Dict
from ..callbacks.callbacks import CallbackHandler
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
        callbacks (CallbackHandler): An object that will handles the callback
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: "optimizer",
        device: torch.device,
        metrics: List["Metric"] = [],
        callbacks: List["Callback"] = [],
    ):
        #  self.model = nn.DataParallel(model).to(device)
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.callbacks = CallbackHandler(metrics + callbacks, self)
        self.device = device
        self.logs = {}

    def fit(
        self, train_dl: "Iterator", n_epochs: int, val_dl: "Iterator" = None
    ):
        self.logs["n_epochs"] = n_epochs
        self.callbacks.on_train_begin(self.logs)
        for i in range(1, n_epochs + 1):
            self.logs["epoch"] = i
            self.fit_epoch(train_dl, val_dl)
            if self.callbacks.on_epoch_end(self.logs):
                break
        self.callbacks.on_train_end(self.logs)

    def fit_epoch(self, train_dl: "Iterator", val_dl: "Iterator"):
        self.model.train()
        self.logs["n_batches"] = len(train_dl)
        self.callbacks.on_epoch_begin(self.logs)
        for i, (train_X, train_y) in enumerate(train_dl, 1):
            train_X = train_X.to(self.device)
            train_y = train_y.to(self.device)
            self.logs["last_X"] = train_X
            self.logs["last_y_true"] = train_y
            self.logs["batch"] = i
            self.callbacks.on_train_batch_begin(self.logs)

            output = self.model(self.logs["last_X"])

            self.logs["last_y_pred"] = output
            self.callbacks.on_loss_begin(self.logs)
            loss = self.criterion(
                self.logs["last_y_pred"], self.logs["last_y_true"]
            )
            self.logs["loss"] = loss.data
            self.callbacks.on_loss_end(self.logs)

            loss.backward()
            self.callbacks.on_step_begin(self.logs)
            self.optimizer.step()
            self.callbacks.on_train_batch_end(self.logs)

        self.model.eval()
        self.logs["n_batches"] = len(val_dl)
        with torch.no_grad():
            for i, (val_X, val_y) in enumerate(val_dl, 1):
                val_X = val_X.to(self.device)
                val_y = val_y.to(self.device)
                self.logs["last_X"] = val_X
                self.logs["last_y_true"] = val_y
                self.logs["batch"] = i
                self.callbacks.on_test_batch_begin(self.logs)

                output = self.model(self.logs["last_X"])

                self.logs["last_y_pred"] = output
                self.callbacks.on_loss_begin(self.logs)
                loss = self.criterion(
                    self.logs["last_y_pred"], self.logs["last_y_true"]
                )
                self.logs["val_loss"] = loss.data
                self.callbacks.on_loss_end(self.logs)
                self.callbacks.on_test_batch_end(self.logs)
