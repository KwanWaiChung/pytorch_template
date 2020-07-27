from ..callbacks import CallbackHandler, Callback, ProgressBar, History, Argmax
from ..utils.logger import getlogger
from .utils.stub.mock_handler import MockLoggingHandler
from collections import namedtuple
import torch

handler = MockLoggingHandler(level="DEBUG")
logger = getlogger()
logger.parent.addHandler(handler)
# 170 -> 101010, good balance of 1 and 0
torch.manual_seed(170)


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
        self.history = History()
        self.callbacks = CallbackHandler([self.history] + callbacks, self)
        self.metrics = []
        self.logs = {}
        self.n_batches = 3
        self.batch_size = 5
        self.use_val = True

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
        self.logs["n_batches"] = self.n_batches
        self.callbacks.on_epoch_begin(self.logs)
        for i in range(self.logs["n_batches"]):
            self.logs["last_X"] = torch.randn(2, 2)
            self.logs["last_y_true"] = torch.randn(2)
            self.logs["batch"] = i
            self.logs["batch_size"] = self.batch_size
            self.callbacks.on_train_batch_begin(self.logs)

            #  output = self.model(self.logs["last_X"])

            self.logs["last_y_pred"] = torch.randn(2)
            self.callbacks.on_loss_begin(self.logs)
            #  loss = self.criterion()
            self.logs["loss"] = (
                ((self.logs["last_y_pred"] - self.logs["last_y_true"]) ** 2)
                .mean()
                .item()
            )
            self.callbacks.on_loss_end(self.logs)

            #  loss.backward()
            self.callbacks.on_step_begin(self.logs)
            #  self.optimizer.step()
            self.callbacks.on_train_batch_end(self.logs)

        # validation
        self.logs["n_batches"] = self.n_batches
        self.callbacks.on_val_begin(self.logs)
        for i in range(self.logs["n_batches"]):
            self.logs["last_X"] = torch.randn(2, 2)
            self.logs["last_y_true"] = torch.randn(1)
            self.logs["batch"] = i
            self.logs["batch_size"] = self.batch_size
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
        self.metrics = []
        self.use_val = True


def test_history():
    n_epochs = 2
    n_batches = 3
    batch_size = 5
    h = History()
    h.set_trainer(DummyTrainer2())
    loss = torch.randn(n_epochs, n_batches)
    val_loss = torch.randn(n_epochs, n_batches)
    for e, (l, val_l) in enumerate(zip(loss, val_loss)):
        h.on_epoch_begin()
        for b, _l in enumerate(l):
            h.on_train_batch_end(
                {
                    "loss": _l.item(),
                    "batch": b,
                    "batch_size": batch_size,
                    "n_batches": n_batches,
                    "epoch": e,
                    "n_epochs": n_epochs,
                }
            )
        h.on_val_begin()
        for b, _val_l in enumerate(val_l):
            h.on_test_batch_end(
                {
                    "val_loss": _val_l.item(),
                    "batch": b,
                    "batch_size": batch_size,
                    "n_batches": n_batches,
                    "epoch": e,
                    "n_epoches": n_epochs,
                }
            )
        h.on_epoch_end({"epoch": e + 1, "n_epochs": n_epochs, "val_loss": 123})
    assert h.epoch == list(range(1, n_epochs + 1))
    assert torch.allclose(torch.tensor(h.history["loss"]), loss.mean(dim=1))
    assert torch.allclose(
        torch.tensor(h.history["val_loss"]), val_loss.mean(dim=1)
    )


def test_callback_basic():
    trainer = DummyTrainer([TestingCallback()])
    trainer.fit(3)


def test_progress_bar():
    trainer = DummyTrainer([ProgressBar()])
    trainer.fit(2)


def test_argmax():
    class ArgmaxTrainer:
        def __init__(self):
            self.callbacks = CallbackHandler([Argmax()], self)
            self.logs = {}

        def fit(self, n_epochs):
            for i in range(1, n_epochs + 1):
                self.fit_epoch()

        def fit_epoch(self):
            # assume one batch only
            self.logs["last_y_pred"] = torch.tensor(
                [[1, 2, 3, 4], [1, 2, 4, 3], [1, 4, 2, 3], [4, 1, 2, 3]]
            )
            self.callbacks.on_loss_end(self.logs)
            assert (
                self.logs["last_y_pred"]
                == torch.tensor([3, 2, 1, 0], dtype=torch.int64)
            ).all()

    trainer = ArgmaxTrainer()
    trainer.fit(1)
