from ..callbacks import (
    CallbackHandler,
    Callback,
    ProgressBar,
    History,
    Argmax,
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
)
from ..trainer import BaseTrainer
from ..metrics import Accuracy
from ..utils.logger import getlogger
from .utils.stub.mock_handler import MockLoggingHandler
from .utils.model import LSTM
from .utils.dataset import getYelpDataloader
from collections import namedtuple
import random
import copy
import torch
import torch.nn as nn
import os
import shutil
import pytest

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


class TestModelCheckpoint:
    def setup_method(self, method):
        torch.manual_seed(170)
        self.lr = 1e-2
        self.handler = MockLoggingHandler(level="DEBUG")
        logger = getlogger()
        logger.parent.addHandler(self.handler)

        self.train_dl, self.val_dl, vocab_size = getYelpDataloader(full=False)
        self.model = LSTM(
            vocab_size=vocab_size,
            embedding_dim=30,
            hidden_dim=10,
            n_layers=1,
            n_classes=2,
        )

    def teardown_method(self, method):
        if os.path.exists("save"):
            shutil.rmtree("save")

    def get_trainer(self, model_checkpoint):
        return BaseTrainer(
            model=self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(self.model.parameters(), lr=self.lr),
            metrics=[Accuracy()],
            callbacks=[Argmax(), model_checkpoint],
        )

    def test_model_checkpoint_always_save(self):
        trainer = self.get_trainer(
            ModelCheckpoint(save_best_only=False, save_weights_only=True)
        )
        trainer.fit(self.train_dl, 2, self.val_dl)
        assert len(os.listdir("save")) == 2
        assert "checkpoint_01_0.69.pth" in os.listdir("save")
        assert "checkpoint_02_0.59.pth" in os.listdir("save")

        state_dict = torch.load(os.path.join("save", "checkpoint_02_0.59.pth"))

        for (k1, v1), (k2, v2) in zip(
            state_dict["model_state_dict"].items(),
            trainer.model.state_dict().items(),
        ):
            assert k1 == k2
            assert (v1 == v2).all()

        # state comparision is too recursive to do
        for (k1, v1), (k2, v2) in zip(
            state_dict["optimizer_state_dict"]["param_groups"][0].items(),
            trainer.optimizer.state_dict()["param_groups"][0].items(),
        ):
            assert k1 == k2
            assert v1 == v2

    def test_model_checkpoint_save_best(self):
        trainer = self.get_trainer(
            ModelCheckpoint(save_best_only=True, save_weights_only=True)
        )
        trainer.fit(self.train_dl, 10, self.val_dl)
        assert len(os.listdir("save")) == 5
        assert "checkpoint_01_0.69.pth" in os.listdir("save")
        assert "checkpoint_02_0.59.pth" in os.listdir("save")
        assert "checkpoint_05_0.34.pth" in os.listdir("save")

        state_dict = torch.load(os.path.join("save", "checkpoint_05_0.34.pth"))
        assert "model_state_dict" in state_dict
        assert "optimizer_state_dict" in state_dict

        # check max epoch == 5
        assert max([int(s.split("_")[1][1]) for s in os.listdir("save")]) == 5

    def test_model_checkpoint_save_model(self):
        trainer = self.get_trainer(
            ModelCheckpoint(save_best_only=False, save_weights_only=False)
        )
        trainer.fit(self.train_dl, 1, self.val_dl)
        assert len(os.listdir("save")) == 1
        assert "checkpoint_01_0.69.pth" in os.listdir("save")

        state_dict = torch.load(os.path.join("save", "checkpoint_01_0.69.pth"))
        assert "model" in state_dict
        assert "model_state_dict" not in state_dict
        assert "optimizer_state_dict" not in state_dict

        model = state_dict["model"]
        trainer = self.get_trainer(
            ModelCheckpoint(save_best_only=False, save_weights_only=False)
        )
        trainer.model = model
        trainer._validate(self.val_dl)
        assert round(trainer.logs["val_loss"], 2) == 0.69

    def test_model_checkpoint_save_weights(self):
        model = copy.deepcopy(self.model)
        trainer = self.get_trainer(
            ModelCheckpoint(save_best_only=False, save_weights_only=True)
        )
        trainer.fit(self.train_dl, 1, self.val_dl)
        assert len(os.listdir("save")) == 1
        assert "checkpoint_01_0.69.pth" in os.listdir("save")
        # check if unequal after trained one epoch
        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), trainer.model.state_dict().items()
        ):
            assert k1 == k2
            assert (v1 != v2).any()

        # load pickle and check if parameters are the same
        state_dict = torch.load(os.path.join("save", "checkpoint_01_0.69.pth"))
        model.load_state_dict(state_dict["model_state_dict"])
        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), trainer.model.state_dict().items()
        ):
            assert k1 == k2
            assert (v1 == v2).all()
        trainer = self.get_trainer(
            ModelCheckpoint(save_best_only=False, save_weights_only=False)
        )
        trainer.model = model
        trainer._validate(self.val_dl)
        assert round(trainer.logs["val_loss"], 2) == 0.69

    def test_model_checkpoint_filepath(self):
        trainer = self.get_trainer(
            ModelCheckpoint(
                filepath="save/checkpoint_{epoch:02d}_{loss:.3f}_"
                "{val_loss:.3f}_{val_acc:.3f}.pth",
                save_best_only=False,
                save_weights_only=True,
            )
        )
        trainer.fit(self.train_dl, 1, self.val_dl)
        assert len(os.listdir("save")) == 1
        assert "checkpoint_01_0.716_0.686_0.471.pth" in os.listdir("save")

    def test_model_checkpoint_with_invalid_filepath(self):
        filepath = "save/checkpoint_{no:02d}"
        trainer = self.get_trainer(
            ModelCheckpoint(
                filepath=filepath, save_best_only=False, save_weights_only=True
            )
        )
        with pytest.raises(KeyError) as e:
            trainer.fit(self.train_dl, 1, self.val_dl)
            assert "Failed to format the filepath: `{}`. "
            "Reason: `{}` is not provided during training".format(
                filepath, "no"
            ) in str(e.value)

    def test_model_checkpoint_restore_training(self):
        filepath = "save/checkpoint_{epoch:02d}_{val_loss:.2f}.pth"
        trainer = self.get_trainer(
            ModelCheckpoint(
                save_best_only=False,
                save_weights_only=True,
                load_weights_on_restart=True,
                filepath=filepath,
            )
        )
        trainer.fit(self.train_dl, 1, self.val_dl)
        assert (
            self.handler.messages["info"][-1]
            == filepath + " is not found. Start training from scratch."
        )
        assert "checkpoint_01_0.69.pth" in os.listdir("save")

        # Resume training
        trainer.fit(self.train_dl, 2, self.val_dl)
        assert (
            self.handler.messages["info"][-1]
            == "Model weights loaded. Resuming training at 2 epoch"
        )
        assert "checkpoint_02_0.59.pth" in os.listdir("save")

    def test_model_checkpoint_restore_training_with_latest_checkpoint(self):
        filepath = "save/checkpoint_{epoch:02d}_{val_loss:.2f}.pth"
        trainer = self.get_trainer(
            ModelCheckpoint(
                save_best_only=False,
                save_weights_only=True,
                load_weights_on_restart=True,
                filepath=filepath,
            )
        )
        trainer.fit(self.train_dl, 2, self.val_dl)

        # Resume training
        trainer.fit(self.train_dl, 3, self.val_dl)
        assert (
            self.handler.messages["info"][-1]
            == "Model weights loaded. Resuming training at 3 epoch"
        )
        assert "checkpoint_03_0.48.pth" in os.listdir("save")

    def test_model_checkpoint_mode(self):
        # pretend that higher val_loss means better for testing
        trainer = self.get_trainer(
            ModelCheckpoint(
                save_best_only=True,
                save_weights_only=True,
                load_weights_on_restart=False,
                monitor="val_loss",
                mode="max",
            )
        )
        trainer.fit(self.train_dl, 7, self.val_dl)

        assert len(os.listdir("save")) == 3
        assert "checkpoint_01_0.69.pth" in os.listdir("save")
        assert "checkpoint_06_0.78.pth" in os.listdir("save")
        assert "checkpoint_07_1.63.pth" in os.listdir("save")

    def test_model_checkpoint_monitor_wrong_metric(self):
        trainer = self.get_trainer(
            ModelCheckpoint(
                save_best_only=False,
                save_weights_only=True,
                load_weights_on_restart=False,
                monitor="non_exist",
                mode="max",
            )
        )

        with pytest.raises(ValueError) as e:
            trainer.fit(self.train_dl, 1, self.val_dl)
            assert "Metric non_exist is not provided during training." in str(
                e.value
            )

    def test_model_checkpoint_with_additional_save_dict(self):
        trainer = self.get_trainer(
            ModelCheckpoint(
                save_best_only=False,
                save_weights_only=True,
                save_dict={"addition": 1},
            )
        )
        trainer.fit(self.train_dl, 1, self.val_dl)
        state_dict = torch.load(os.path.join("save", "checkpoint_01_0.69.pth"))
        assert state_dict["addition"] == 1


class TestEarlyStopping:
    def setup_method(self, method):
        os.makedirs("save", exist_ok=True)
        torch.manual_seed(170)
        self.lr = 1e-2
        self.handler = MockLoggingHandler(level="DEBUG")
        logger = getlogger()
        logger.parent.addHandler(self.handler)

        self.train_dl, self.val_dl, vocab_size = getYelpDataloader(full=False)
        self.model = LSTM(
            vocab_size=vocab_size,
            embedding_dim=30,
            hidden_dim=10,
            n_layers=1,
            n_classes=2,
        )

    def teardown_method(self, method):
        if os.path.exists("save"):
            shutil.rmtree("save")

    def get_trainer(self, callbacks):
        return BaseTrainer(
            model=self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(self.model.parameters(), lr=self.lr),
            metrics=[Accuracy()],
            callbacks=[Argmax()] + callbacks,
        )

    def test_early_stopping_patience(self):
        trainer = self.get_trainer([EarlyStopping(patience=2)])
        trainer.fit(self.train_dl, 10, self.val_dl)
        assert len(trainer.history.history["val_loss"]) == 7

    def test_early_stopping_pickle(self):
        callback = EarlyStopping(patience=5)
        callback.on_train_begin({})
        callback.best_score = 0.5
        callback.patience = 2
        torch.save({"c": callback}, "save/save.pth")
        c = torch.load("save/save.pth")
        assert c["c"].best_score == 0.5
        assert c["c"].patience == 2

    def test_early_stopping_model_checkpoint_integration(self):
        early_stop = EarlyStopping(patience=3)
        model_checkpoint = ModelCheckpoint(
            save_best_only=False,
            save_weights_only=True,
            save_dict={"early_stopping": early_stop},
            load_weights_on_restart=True,
        )
        trainer = self.get_trainer([early_stop, model_checkpoint])
        trainer.fit(self.train_dl, 7, self.val_dl)

        # resume training
        early_stop = EarlyStopping(patience=3)
        model_checkpoint = ModelCheckpoint(
            save_best_only=False,
            save_weights_only=True,
            save_dict={"early_stopping": early_stop},
            load_weights_on_restart=True,
        )
        trainer = self.get_trainer([early_stop, model_checkpoint])
        model_checkpoint.on_train_begin({})
        for value in trainer.callbacks.callbacks:
            if isinstance(value, EarlyStopping):
                assert value.wait == 3
                assert abs(value.best_score - 0.342) < 1e-3

        trainer.fit(self.train_dl, 10, self.val_dl)
        assert trainer.history.epoch[-1] == 8


class TestTensorboard:
    def test_tensorboard(self):
        tensorBoard = TensorBoard("logs")
        metric = namedtuple("Metric", ["name"])
        trainer = namedtuple("Trainer", ["use_val", "metrics"])(
            True, [metric("f1")]
        )
        tensorBoard.set_trainer(trainer)

        for i in range(1, 21):
            tensorBoard.on_epoch_end(
                {
                    "loss": random.random(),
                    "val_loss": random.random(),
                    "f1": random.random(),
                    "val_f1": random.random(),
                    "epoch": i,
                }
            )
        if os.path.exists("logs"):
            shutil.rmtree("logs")
