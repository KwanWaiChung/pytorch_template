from ..callbacks.callbacks import (
    CallbackHandler,
    Callback,
    ProgressBar,
    History,
)
from ..utils.logger import getlogger
from .mock_handler import MockLoggingHandler
from collections import namedtuple
import torch

handler = MockLoggingHandler(level="DEBUG")
logger = getlogger()
logger.parent.addHandler(handler)


class TestingCallback(Callback):
    def on_train_begin(self, logs):
        logger.info("begin training")
        assert "n_epochs" in logs

    def on_train_end(self, logs):
        logger.info("end training")
        assert "n_epochs" in logs

    def on_epoch_begin(self, logs):
        logger.info("begin epoch")
        assert "n_epochs" in logs
        assert "epoch" in logs

    def on_epoch_end(self, logs):
        logger.info("end epoch")
        assert "n_epochs" in logs
        assert "epoch" in logs
        return False

    def on_loss_begin(self, logs):
        logger.info("begin loss")
        assert "last_y_pred" in logs
        assert "n_epochs" in logs
        assert "epoch" in logs
        assert "last_X" in logs
        assert "last_y_true" in logs
        assert "batch" in logs
        assert "n_batches" in logs

    def on_loss_end(self, logs):
        logger.info("end loss")
        assert "loss" in logs

    def on_step_begin(self, logs):
        logger.info("begin step")
        assert "loss" in logs
        assert "last_y_pred" in logs
        assert "n_epochs" in logs
        assert "epoch" in logs
        assert "last_X" in logs
        assert "last_y_true" in logs
        assert "batch" in logs
        assert "n_batches" in logs

    def on_train_batch_begin(self, logs):
        logger.info("begin train batch")
        assert "n_epochs" in logs
        assert "epoch" in logs
        assert "last_X" in logs
        assert "last_y_true" in logs
        assert "batch" in logs
        assert "n_batches" in logs

    def on_train_batch_end(self, logs):
        logger.info("end train batch")

    def on_test_batch_begin(self, logs):
        logger.info("begin test batch")
        assert "n_epochs" in logs
        assert "epoch" in logs
        assert "last_X" in logs
        assert "last_y_true" in logs
        assert "batch" in logs
        assert "n_batches" in logs

    def on_test_batch_end(self, logs):
        logger.info("end test batch")
        assert "last_y_pred" in logs
        assert "val_loss" in logs


class DummyTrainer:
    def __init__(self, callbacks):
        self.callbacks = CallbackHandler(callbacks, self)
        self.metrics = []
        self.logs = {}

    def fit(self, n_epochs):
        self.logs["n_epochs"] = n_epochs
        self.callbacks.on_train_begin(self.logs)
        for i in range(1, n_epochs + 1):
            self.logs["epoch"] = i
            self.fit_epoch()
            if self.callbacks.on_epoch_end(self.logs):
                break
        self.callbacks.on_train_end(self.logs)

    def fit_epoch(self):
        #  self.model.train()
        self.logs["n_batches"] = 3
        self.callbacks.on_epoch_begin(self.logs)
        for i in range(self.logs["n_batches"]):
            self.logs["last_X"] = torch.randn(2, 2)
            self.logs["last_y_true"] = torch.randn(1)
            self.logs["batch"] = i
            self.callbacks.on_train_batch_begin(self.logs)

            #  output = self.model(self.logs["last_X"])

            self.logs["last_y_pred"] = torch.randn(1)
            self.callbacks.on_loss_begin(self.logs)
            #  loss = self.criterion()
            self.logs["loss"] = (
                (self.logs["last_y_pred"] - self.logs["last_y_true"]) ** 2
            ).item()
            self.callbacks.on_loss_end(self.logs)

            #  loss.backward()
            self.callbacks.on_step_begin(self.logs)
            #  self.optimizer.step()
            self.callbacks.on_train_batch_end(self.logs)

        # validation
        self.logs["n_batches"] = 2
        for i in range(self.logs["n_batches"]):
            self.logs["last_X"] = torch.randn(2, 2)
            self.logs["last_y_true"] = torch.randn(1)
            self.logs["batch"] = 2
            self.callbacks.on_test_batch_begin(self.logs)

            #  output = self.model(self.logs["last_X"])

            self.logs["last_y_pred"] = torch.randn(1)

            #  loss = self.criterion()

            self.logs["val_loss"] = (
                (self.logs["last_y_pred"] - self.logs["last_y_true"]) ** 2
            ).item()
            self.callbacks.on_test_batch_end(self.logs)


class DummyTrainer2:
    metric = namedtuple("Metric", ["name"])

    def __init__(self):
        self.metrics = [self.metric("f1")]


def test_callback_basic():
    trainer = DummyTrainer([TestingCallback()])
    trainer.fit(3)


def test_progress_bar():
    trainer = DummyTrainer([ProgressBar()])
    trainer.fit(2)


def test_history():
    h = History()
    h.set_trainer(DummyTrainer2())
    f1s = torch.randn(5)
    loss = torch.randn(5)
    val_loss = torch.randn(5)
    val_f1s = torch.randn(5)
    for i, (f1, l, val_l, val_f1) in enumerate(
        zip(f1s, loss, val_loss, val_f1s)
    ):
        h.on_epoch_end(
            {
                "epoch": i + 1,
                "loss": l.item(),
                "val_loss": val_l.item(),
                "f1": f1.item(),
                "val_f1": val_f1.item(),
            }
        )
    assert h.epoch == list(range(1, 6))
    assert h.history["loss"] == loss.tolist()
    assert h.history["val_loss"] == val_loss.tolist()
    assert h.history["f1"] == f1s.tolist()
    assert h.history["val_f1"] == val_f1s.tolist()
