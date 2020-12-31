from typing import List, Union, Callable, Dict, Iterable, Any
from copy import deepcopy
from ..callbacks import (
    CallbackHandler,
    History,
    ProgressBar,
    GpuTrainer,
    ModelCheckpoint,
    EarlyStopping,
)
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
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
        metrics: List["Metric"] = [],
        callbacks: List["Callback"] = [],
        model_checkpoint: ModelCheckpoint = None,
        early_stopping: EarlyStopping = None,
        gpu_id: int = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        post_callbacks = []
        if early_stopping:
            post_callbacks.append(early_stopping)
        if model_checkpoint:
            post_callbacks.append(model_checkpoint)
        post_callbacks.append(ProgressBar())
        self.history = History()
        self.callbacks = CallbackHandler(
            [GpuTrainer(gpu_id)]
            + metrics
            + [self.history]
            + callbacks
            + post_callbacks,
            self,
        )
        self.use_val = False
        self.logs = {}

    def training_step(
        self, batch_dict: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """Define a training step of the model.

        Args:
            batch_dict: Input for the model.
            batch_idx: Current batch idx of this epoch.

        Returns:
            Dict with required keys:
                loss (torch.Tensor): The avg loss of current batch.
                batch_size (int): The batch size of current batch.
                y_pred (torch.Tensor): The predicted tensor.
                y_true (torch.Tensor): The ground truth tensor.

        """
        raise NotImplementedError

    def validation_step(self, batch_dict, batch_idx):
        """Define a validation step of the model.

        Args:
            batch_dict: Input for the model.
            batch_idx: Current batch idx of this epoch.

        Returns:
            Dict with required keys:
                val_loss (torch.Tensor): The avg loss of current batch.
                batch_size (int): The batch size of current batch.
                y_pred (torch.Tensor): The predicted tensor.
                y_true (torch.Tensor): The ground truth tensor.

        """
        raise NotImplementedError

    def testing_step(self, batch_dict, batch_idx):
        """Define a testing step of the model.

        Args:
            batch_dict: Input for the model.
            batch_idx: Current batch idx of this epoch.

        Returns:
            Dict with required keys:
                test_loss (torch.Tensor): The avg loss of current batch.
                batch_size (int): The batch size of current batch.
                y_pred (torch.Tensor): The predicted tensor.
                y_true (torch.Tensor): The ground truth tensor.
                    Optional, you should pass it if some metrics need it.

        """
        raise NotImplementedError

    def _training_step(
        self, batch_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        outputs = self.training_step(batch_dict, self.logs["batch_idx"])
        for key in ["loss", "batch_size", "y_pred", "y_true"]:
            if key not in outputs:
                raise KeyError(
                    f"`{key}` should be included in the returned dict "
                    "of `training_step()`"
                )
        outputs["y_pred"] = outputs["y_pred"].detach().cpu()
        outputs["y_true"] = outputs["y_true"].detach().cpu()
        return outputs

    def _validation_step(
        self, batch_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        outputs = self.validation_step(batch_dict, self.logs["batch_idx"])
        for key in ["val_loss", "batch_size", "y_pred", "y_true"]:
            if key not in outputs:
                raise KeyError(
                    f"`{key}` should be included in the returned dict "
                    "of `validation_step()`"
                )
        outputs["val_loss"] = outputs["val_loss"].item()
        outputs["y_pred"] = outputs["y_pred"].detach().cpu()
        outputs["y_true"] = outputs["y_true"].detach().cpu()
        return outputs

    def _testing_step(
        self,
        batch_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        outputs = self.testing_step(batch_dict, self.logs["batch_idx"])
        for key in ["test_loss", "batch_size", "y_pred"]:
            if key not in outputs:
                raise KeyError(
                    f"`{key}` should be included in the returned dict "
                    "of `testing_step()`"
                )
        outputs["test_loss"] = outputs["test_loss"].item()
        outputs["y_pred"] = outputs["y_pred"].detach().cpu()
        if "y_true" in outputs:
            outputs["y_true"] = outputs["y_true"].detach().cpu()
        return outputs

    def fit(self, train_dl: Iterable, n_epochs: int, val_dl: Iterable = None):
        if val_dl is not None:
            self.use_val = True
        self.logs["n_epochs"] = n_epochs
        self.callbacks.on_train_begin(self.logs)
        for i in range(self.logs.get("epoch_idx", 1), n_epochs + 1):
            self.logs["epoch_idx"] = i
            self.fit_epoch(train_dl, val_dl)
            if self.callbacks.on_epoch_end(self.logs):
                break
        self.callbacks.on_train_end(self.logs)

    def fit_epoch(self, train_dl: Iterable, val_dl: Iterable):
        """Run one training epoch.

        Returns:
            bool: True, if early stop after trained certain batches.
        """
        self.logs["n_batches"] = len(train_dl)
        self.callbacks.on_epoch_begin(self.logs)
        self._train(train_dl)
        if self.use_val:
            self._validate(val_dl)

    def _train_batch(self, batch_dict: Dict, batch_idx: int, tracking=True):
        """
        Returns:
            loss (float): The loss of current batch
            tracking (bool): Enable `on_train_batch_end` for comprehensive
                logging
        """
        self.logs["batch_dict"] = batch_dict
        self.logs["batch_idx"] = batch_idx
        self.callbacks.on_train_batch_begin(self.logs)

        outputs = self._training_step(self.logs["batch_dict"])
        loss = outputs["loss"]
        self.logs.update(outputs)
        self.callbacks.on_loss_end(self.logs)

        self.optimizer.zero_grad()
        loss.backward()
        self.callbacks.on_step_begin(self.logs)
        self.optimizer.step()
        if tracking:
            self.callbacks.on_train_batch_end(self.logs)
        return loss.item()

    def _train(self, train_dl):
        """Training for one epoch.

        Returns:
            bool: True, if early stop after trained certain batches.

        """
        self.model.train()
        for i, batch_dict in enumerate(train_dl, 1):
            self._train_batch(batch_dict, i)

    def _validate(self, val_dl):
        self.model.eval()
        self.logs["n_batches"] = len(val_dl)
        self.callbacks.on_val_begin(self.logs)
        with torch.no_grad():
            for i, batch_dict in enumerate(val_dl, 1):
                self.logs["batch_dict"] = batch_dict
                self.logs["batch_idx"] = i
                self.callbacks.on_val_batch_begin(self.logs)

                outputs = self._validation_step(self.logs["batch_dict"])
                self.logs.update(outputs)
                self.callbacks.on_val_batch_end(self.logs)
        self.callbacks.on_val_end(self.logs)

    def test(self, test_dl):
        self.model.eval()
        self.logs = {}
        self.logs["n_batches"] = len(test_dl)
        self.callbacks.on_test_begin(self.logs)
        with torch.no_grad():
            for i, batch_dict in enumerate(test_dl, 1):
                self.logs["batch_dict"] = batch_dict
                self.logs["batch_idx"] = i
                self.callbacks.on_test_batch_begin(self.logs)

                outputs = self._testing_step(self.logs["batch_dict"])
                self.logs.update(outputs)
                self.callbacks.on_test_batch_end(self.logs)
        self.callbacks.on_test_end(self.logs)

    def get_best_filepath(self):
        for cb in self.callbacks.callbacks:
            if isinstance(cb, ModelCheckpoint):
                return cb.best_filepath
        return None

    def state_dict(self):
        """Get all the related state dicts for saving.
        Will be called from model_checkpoint callback.

        Returns:
            dict
        """
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state_dict(self, d):
        self.model.load_state_dict(d["model_state_dict"])
        self.optimizer.load_state_dict(d["optimizer_state_dict"])
