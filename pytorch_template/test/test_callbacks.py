from ..callbacks import (
    CallbackHandler,
    Callback,
    ProgressBar,
    History,
    ModelCheckpoint,
    EarlyStopping,
    Wandblogger,
)
from ..trainer import BaseTrainer
from ..metrics import Accuracy
from ..utils.logger import getlogger
from .utils.stub import (
    RegressionTrainer,
    ClassificationTrainer,
    MockLoggingHandler,
)
from .utils.model import LSTM
from .utils.dataset import getYelpDataloader
from ..utils.misc import set_seed
from collections import namedtuple
import random
import copy
import torch
import torch.nn as nn
import os
import shutil
import pytest
from time import sleep

handler = MockLoggingHandler(level="DEBUG")
logger = getlogger()
logger.parent.addHandler(handler)
# 170 -> 101010, good balance of 1 and 0
torch.manual_seed(170)


class SimpleCallback(Callback):
    def __init__(self, metric: str = None):
        super().__init__()
        self.metric = metric

    def on_train_begin(self, logs):
        logger.info("begin training")
        assert "n_epochs" in logs
        assert type(logs["n_epochs"]) == int

    def on_epoch_begin(self, logs):
        logger.info("begin epoch")
        assert "epoch_idx" in logs
        assert type(logs["epoch_idx"]) == int
        assert "n_batches" in logs
        assert type(logs["n_batches"]) == int

    def on_train_batch_begin(self, logs):
        logger.info("begin train batch")
        assert "batch_dict" in logs
        assert type(logs["batch_dict"]) == dict
        assert "batch_idx" in logs
        assert type(logs["batch_idx"]) == int

    def on_loss_end(self, logs):
        logger.info("end loss")
        assert "loss" in logs
        assert type(logs["loss"]) == float
        assert "y_pred" in logs
        assert type(logs["y_pred"]) == torch.Tensor
        assert "y_true" in logs
        assert type(logs["y_true"]) == torch.Tensor
        assert "batch_size" in logs
        assert type(logs["batch_size"]) == int

    def on_step_begin(self, logs):
        logger.info("begin step")

    def on_train_batch_end(self, logs):
        logger.info("end train batch")
        if self.metric:
            assert self.metric in logs
            assert type(logs[self.metric]) == float

    def on_val_begin(self, logs):
        logger.info("begin validating")
        assert "epoch_idx" in logs
        assert type(logs["epoch_idx"]) == int
        assert "n_epochs" in logs
        assert type(logs["n_epochs"]) == int
        assert "n_batches" in logs
        assert type(logs["n_batches"]) == int

    def on_val_batch_begin(self, logs):
        logger.info("begin val batch")
        assert "batch_dict" in logs
        assert type(logs["batch_dict"]) == dict
        assert "batch_idx" in logs
        assert type(logs["batch_idx"]) == int
        assert "n_batches" in logs
        assert type(logs["n_batches"]) == int
        assert "n_epochs" in logs
        assert type(logs["n_epochs"]) == int
        assert "batch_size" in logs
        assert type(logs["batch_size"]) == int
        assert "epoch_idx" in logs
        assert type(logs["epoch_idx"]) == int

    def on_val_batch_end(self, logs):
        logger.info("end val batch")
        if self.metric:
            assert f"val_{self.metric}" in logs
            assert type(logs[f"val_{self.metric}"]) == float
        assert "val_loss" in logs
        assert type(logs["val_loss"]) == float
        assert "y_pred" in logs
        assert type(logs["y_pred"]) == torch.Tensor
        assert "y_true" in logs
        assert type(logs["y_true"]) == torch.Tensor

    def on_val_end(self, logs):
        logger.info("end val")

    def on_epoch_end(self, logs):
        logger.info("end epoch")
        assert "val_loss" in logs
        assert "loss" in logs
        assert "n_epochs" in logs
        assert "epoch_idx" in logs
        return False

    def on_train_end(self, logs):
        logger.info("end training")
        assert "n_epochs" in logs
        assert type(logs["n_epochs"]) == int


class DummyTrainer2:
    metric = namedtuple("Metric", ["name"])

    def __init__(self):
        self.metrics = [self.metric("acc")]
        self.use_val = True


def test_history():
    set_seed(1337)
    n_epochs = 2
    n_batches = 3
    batch_size = 5
    h = History()
    h.set_trainer(DummyTrainer2())
    loss = torch.randn(n_epochs, n_batches)
    val_loss = torch.randn(n_epochs, n_batches)
    acc = torch.randn(n_epochs, n_batches)
    val_acc = torch.randn(n_epochs, n_batches)
    for e, (l, val_l, a, val_a) in enumerate(
        zip(loss, val_loss, acc, val_acc)
    ):
        h.on_epoch_begin()
        for b, (_l, _a) in enumerate(zip(l, a)):
            h.on_train_batch_end(
                {
                    "acc": _a.item(),
                    "loss": _l.item(),
                    "batch_idx": b,
                    "batch_size": batch_size,
                    "n_batches": n_batches,
                    "epoch": e,
                    "n_epochs": n_epochs,
                }
            )
        h.on_val_begin()
        for b, (_val_l, _val_a) in enumerate(zip(val_l, val_a)):
            h.on_val_batch_end(
                {
                    "val_acc": _val_a.item(),
                    "val_loss": _val_l.item(),
                    "batch_idx": b,
                    "batch_size": batch_size,
                    "n_batches": n_batches,
                    "epoch": e,
                    "n_epoches": n_epochs,
                }
            )
        h.on_epoch_end(
            {
                "epoch_idx": e + 1,
                "n_epochs": n_epochs,
                "acc": acc[-1, -1],
                "val_acc": val_acc[-1, -1],
            }
        )
    assert h.epoch == list(range(1, n_epochs + 1))
    # rolling average is done in History
    assert torch.allclose(torch.tensor(h.history["loss"]), loss.mean(dim=1))
    assert torch.allclose(
        torch.tensor(h.history["val_loss"]), val_loss.mean(dim=1)
    )
    # metric average is done in Metric, History will just take the value
    assert torch.allclose(torch.tensor(h.history["acc"]), acc[-1, -1])
    assert torch.allclose(torch.tensor(h.history["val_acc"]), val_acc[-1, -1])


def test_callback_basic():
    set_seed(170)
    lr = 1e-2
    train_dl, val_dl, vocab_size = getYelpDataloader()

    model = LSTM(
        vocab_size=vocab_size,
        embedding_dim=30,
        hidden_dim=10,
        n_layers=1,
        n_classes=2,
    )
    trainer = ClassificationTrainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        callbacks=[SimpleCallback()],
        metrics=[Accuracy()],
    )
    trainer.fit(train_dl=train_dl, n_epochs=3, val_dl=val_dl)
    if os.path.exists("saved"):
        shutil.rmtree("saved")


class TestModelCheckpoint:
    def setup_method(self, method):
        set_seed(170)
        self.lr = 1e-2
        self.handler = MockLoggingHandler(level="DEBUG")
        logger = getlogger()
        logger.parent.addHandler(self.handler)

        self.train_dl, self.val_dl, vocab_size = getYelpDataloader()
        self.model = LSTM(
            vocab_size=vocab_size,
            embedding_dim=30,
            hidden_dim=10,
            n_layers=1,
            n_classes=2,
        )

    def teardown_method(self, method):
        if os.path.exists("saved"):
            shutil.rmtree("saved")

    def get_trainer(self, model_checkpoint):
        return ClassificationTrainer(
            model=self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(self.model.parameters(), lr=self.lr),
            model_checkpoint=model_checkpoint,
            metrics=[Accuracy()],
        )

    def test_model_checkpoint_always_save(self):
        trainer = self.get_trainer(
            ModelCheckpoint(save_best_only=False, save_weights_only=True)
        )
        trainer.fit(self.train_dl, 2, self.val_dl)
        assert len(os.listdir("saved/models")) == 2
        assert "checkpoint_01_0.6704.pth" in os.listdir("saved/models")
        assert "checkpoint_02_0.6076.pth" in os.listdir("saved/models")

        state_dict = torch.load(
            os.path.join("saved/models", "checkpoint_02_0.6076.pth")
        )["model_checkpoint"]

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
        ckpt = ModelCheckpoint(save_best_only=True, save_weights_only=True)
        trainer = self.get_trainer(ckpt)
        trainer.fit(self.train_dl, 20, self.val_dl)
        assert len(os.listdir("saved/models")) == 5
        assert "checkpoint_01_0.6704.pth" in os.listdir("saved/models")
        assert "checkpoint_02_0.6076.pth" in os.listdir("saved/models")
        assert "checkpoint_03_0.5325.pth" in os.listdir("saved/models")
        assert "checkpoint_04_0.2953.pth" in os.listdir("saved/models")
        assert "last.pth" in os.listdir("saved/models")
        assert ckpt.best_filepath == "saved/models/checkpoint_04_0.2953.pth"

        state_dict = torch.load(
            os.path.join("saved/models", "checkpoint_02_0.6076.pth")
        )["model_checkpoint"]
        assert "model_state_dict" in state_dict
        assert "optimizer_state_dict" in state_dict

    def test_model_checkpoint_max_save(self):
        ckpt = ModelCheckpoint(max_save=2)
        trainer = self.get_trainer(ckpt)
        trainer.fit(self.train_dl, 4, self.val_dl)
        assert len(os.listdir("saved/models")) == 2
        assert (
            self.handler.messages["info"][-1] == "Removed oldest checkpoint: "
            "saved/models/checkpoint_02_0.6076.pth"
        )
        assert (
            self.handler.messages["info"][-2] == "Removed oldest checkpoint: "
            "saved/models/checkpoint_01_0.6704.pth"
        )
        assert "checkpoint_03_0.5325.pth" in os.listdir("saved/models")
        assert "checkpoint_04_0.2953.pth" in os.listdir("saved/models")

    def test_model_checkpoint_save_model(self):
        trainer = self.get_trainer(
            ModelCheckpoint(save_best_only=False, save_weights_only=False)
        )
        trainer.fit(self.train_dl, 1, self.val_dl)
        assert len(os.listdir("saved/models")) == 1
        assert "checkpoint_01_0.6704.pth" in os.listdir("saved/models")

        state_dict = torch.load(
            os.path.join("saved/models", "checkpoint_01_0.6704.pth")
        )["model_checkpoint"]
        assert "model" in state_dict
        assert "model_state_dict" not in state_dict
        assert "optimizer_state_dict" not in state_dict

        model = state_dict["model"]
        trainer = self.get_trainer(
            ModelCheckpoint(save_best_only=False, save_weights_only=False)
        )
        trainer.model = model
        trainer._validate(self.val_dl)
        assert round(trainer.logs["val_loss"], 4) == 0.6704

    def test_model_checkpoint_save_weights(self):
        model = copy.deepcopy(self.model)
        trainer = self.get_trainer(
            ModelCheckpoint(save_best_only=False, save_weights_only=True)
        )
        trainer.fit(self.train_dl, 1, self.val_dl)
        assert len(os.listdir("saved/models")) == 1
        assert "checkpoint_01_0.6704.pth" in os.listdir("saved/models")
        # check if unequal after trained one epoch
        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), trainer.model.state_dict().items()
        ):
            assert k1 == k2
            assert (v1 != v2).any()

        # load pickle and check if parameters are the same
        state_dict = torch.load(
            os.path.join("saved/models", "checkpoint_01_0.6704.pth")
        )["model_checkpoint"]
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
        assert round(trainer.logs["val_loss"], 4) == 0.6704

    def test_model_checkpoint_filepath(self):
        trainer = self.get_trainer(
            ModelCheckpoint(
                filepath="saved/models/checkpoint_{epoch_idx:02d}_{loss:.4f}_"
                "{val_loss:.4f}_{val_acc:.4f}.pth",
                save_best_only=False,
                save_weights_only=True,
            )
        )
        trainer.fit(self.train_dl, 1, self.val_dl)
        assert len(os.listdir("saved/models")) == 1
        assert "checkpoint_01_0.6803_0.6704_0.6667.pth" in os.listdir(
            "saved/models"
        )

    def test_model_checkpoint_with_invalid_filepath(self):
        filepath = "saved/models/checkpoint_{no:02d}"
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

    def test_model_checkpoint_restore_training_save_weights(self):
        filepath = "saved/models/checkpoint_{epoch_idx:02d}_{val_loss:.4f}.pth"
        #  filepath = "save/last.pth"
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
            == f"{filepath} is not found. Start training from scratch."
        )
        assert "checkpoint_01_0.6704.pth" in os.listdir("saved/models")

        # Resume training
        trainer.fit(self.train_dl, 2, self.val_dl)
        assert (
            self.handler.messages["info"][-1]
            == "Model weights loaded from saved/models/checkpoint_01_0.6704.pth. "
            "Resuming training at 2 epoch"
        )
        assert "checkpoint_02_0.6076.pth" in os.listdir("saved/models")

    def test_model_checkpoint_restore_training_save_best_weights(self):
        #  filepath = "save/last.pth"
        ckpt = ModelCheckpoint(
            save_best_only=True,
            save_weights_only=True,
            load_weights_on_restart=True,
        )
        trainer = self.get_trainer(ckpt)
        trainer.fit(self.train_dl, 1, self.val_dl)
        assert (
            self.handler.messages["info"][-1]
            == "saved/models/last.pth is not found. Start training from scratch."
        )
        assert "checkpoint_01_0.6704.pth" in os.listdir("saved/models")
        assert "last.pth" in os.listdir("saved/models")
        modtime = os.path.getmtime("saved/models/last.pth")

        ckpt = ModelCheckpoint(
            save_best_only=True,
            save_weights_only=True,
            load_weights_on_restart=True,
        )
        trainer = self.get_trainer(ckpt)
        ckpt.on_train_begin({})
        for value in trainer.callbacks.callbacks:
            if isinstance(value, ModelCheckpoint):
                assert (
                    value.best_filepath
                    == "saved/models/checkpoint_01_0.6704.pth"
                )
        # Resume training
        sleep(1)
        trainer.fit(self.train_dl, 2, self.val_dl)
        assert (
            self.handler.messages["info"][-1]
            == "Model weights loaded from saved/models/last.pth. "
            "Resuming training at 2 epoch"
        )
        assert "checkpoint_02_0.6076.pth" in os.listdir("saved/models")
        assert os.path.getmtime("saved/models/last.pth") > modtime

    def test_model_checkpoint_restore_training_with_latest_checkpoint(self):
        trainer = self.get_trainer(
            ModelCheckpoint(
                save_best_only=False,
                save_weights_only=True,
                load_weights_on_restart=True,
            )
        )
        trainer.delay = True
        trainer.fit(self.train_dl, 2, self.val_dl)

        # Resume training
        trainer.fit(self.train_dl, 3, self.val_dl)
        assert (
            self.handler.messages["info"][-1]
            == "Model weights loaded from saved/models/checkpoint_02_0.6076.pth. "
            "Resuming training at 3 epoch"
        )
        assert "checkpoint_03_0.5325.pth" in os.listdir("saved/models")

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
        assert len(os.listdir("saved/models")) == 3
        assert "checkpoint_01_0.6704.pth" in os.listdir("saved/models")
        assert "checkpoint_07_0.8819.pth" in os.listdir("saved/models")
        assert "last.pth" in os.listdir("saved/models")

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


class TestEarlyStopping:
    def setup_method(self, method):
        set_seed(170)
        self.lr = 1e-2
        self.handler = MockLoggingHandler(level="DEBUG")
        logger = getlogger()
        logger.parent.addHandler(self.handler)

        self.train_dl, self.val_dl, vocab_size = getYelpDataloader()
        self.model = LSTM(
            vocab_size=vocab_size,
            embedding_dim=30,
            hidden_dim=10,
            n_layers=1,
            n_classes=2,
        )

    def teardown_method(self, method):
        if os.path.exists("saved/models"):
            shutil.rmtree("saved/models")

    def get_trainer(self, early_stopping, model_checkpoint=None):
        return ClassificationTrainer(
            model=self.model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(self.model.parameters(), lr=self.lr),
            early_stopping=early_stopping,
            model_checkpoint=model_checkpoint,
            metrics=[Accuracy()],
        )

    def test_early_stopping_patience(self):
        trainer = self.get_trainer(EarlyStopping(patience=2))
        trainer.fit(self.train_dl, 10, self.val_dl)
        for value in trainer.callbacks.callbacks:
            if isinstance(value, EarlyStopping):
                assert value.wait == 2
                assert abs(value.best_score - 0.2953) < 1e-4
        assert len(trainer.history.history["val_loss"]) == 7

    def test_early_stopping_pickle(self):
        callback = EarlyStopping(patience=5)
        callback.on_train_begin({})
        callback.best_score = 0.5
        callback.patience = 2
        os.makedirs("saved/models", exist_ok=True)
        torch.save({"c": callback}, "saved/models/save.pth")
        c = torch.load("saved/models/save.pth")
        assert c["c"].best_score == 0.5
        assert c["c"].patience == 2

    def test_early_stopping_model_checkpoint_integration(self):
        early_stopping = EarlyStopping(patience=4)
        model_checkpoint = ModelCheckpoint(
            save_best_only=False,
            save_weights_only=True,
            load_weights_on_restart=True,
        )
        trainer = self.get_trainer(
            early_stopping=early_stopping, model_checkpoint=model_checkpoint
        )
        trainer.delay = True
        trainer.fit(self.train_dl, 7, self.val_dl)
        for value in trainer.callbacks.callbacks:
            if isinstance(value, EarlyStopping):
                assert value.wait == 3
                assert abs(value.best_score - 0.2953) < 1e-4

        # resume
        early_stop = EarlyStopping(patience=5)
        model_checkpoint = ModelCheckpoint(
            save_best_only=False,
            save_weights_only=True,
            load_weights_on_restart=True,
        )
        trainer = self.get_trainer(
            early_stopping=early_stopping, model_checkpoint=model_checkpoint
        )
        model_checkpoint.on_train_begin({})
        for value in trainer.callbacks.callbacks:
            if isinstance(value, EarlyStopping):
                assert value.wait == 3
                assert abs(value.best_score - 0.2953) < 1e-4

        trainer.fit(self.train_dl, 10, self.val_dl)
        assert trainer.history.epoch[-1] == 9


#  class TestTrainLogger:
#      def setup_method(self, method):
#          set_seed(170)
#          self.lr = 1e-2
#          self.handler = MockLoggingHandler(level="DEBUG")
#          logger = getlogger()
#          logger.parent.addHandler(self.handler)

#          self.train_dl, self.val_dl, vocab_size = getYelpDataloader()
#          self.model = LSTM(
#              vocab_size=vocab_size,
#              embedding_dim=30,
#              hidden_dim=10,
#              n_layers=1,
#              n_classes=2,
#          )

#      def get_trainer(self, callbacks):
#          return ClassificationTrainer(
#              model=self.model,
#              criterion=nn.CrossEntropyLoss(),
#              optimizer=torch.optim.Adam(self.model.parameters(), lr=self.lr),
#              callbacks=callbacks,
#              metrics=[Accuracy()],
#              model_checkpoint=ModelCheckpoint(save_best_only=True),
#          )

#      # disable this test as it will create the ckpt which
#      # we can't delete for this test case
#      def test_wandb(self):
#          logger = Wandblogger(
#              "Testing", "testin123", config={"param1": 100, "param2": 200}
#          )
#          trainer = self.get_trainer([logger])
#          trainer.delay = True
#          trainer.fit(self.train_dl, 7, self.val_dl)
#          trainer.test(self.train_dl)
