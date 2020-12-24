import os
import torch
import shutil
import copy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import LRFinder, getlogger
from .utils.stub import MockLoggingHandler
from .utils.model import LSTM
from .utils.dataset import getYelpDataloader
from .utils.stub import RegressionTrainer, ClassificationTrainer
from ..trainer import BaseTrainer
from ..metrics import Accuracy


class TestLRFinder:
    def setup_method(self, method):
        torch.manual_seed(170)
        self.handler = MockLoggingHandler(level="DEBUG")
        logger = getlogger()
        logger.parent.addHandler(self.handler)

        self.train_dl, self.val_dl, vocab_size = getYelpDataloader()
        model = LSTM(
            vocab_size=vocab_size,
            embedding_dim=30,
            hidden_dim=10,
            n_layers=1,
            n_classes=2,
        )
        self.trainer = ClassificationTrainer(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            metrics=[Accuracy()],
        )

    def teardown_method(self, method):
        if os.path.exists("plot"):
            shutil.rmtree("plot")

    def test_exponential_scheduling_with_plot(self):
        model_state = copy.deepcopy(self.trainer.model.state_dict())
        optimizer_state = copy.deepcopy(self.trainer.optimizer.state_dict())

        lr_finder = LRFinder(self.trainer)
        #  lr_finder.fit(self.train_dl, self.val_dl, step_mode="exp")
        lr_finder.fit(self.train_dl, self.val_dl, step_mode="exp")
        assert (
            self.handler.messages["info"][-1]
            == "Learning rate search has finished. Model and "
            "optimizer weights has been restored."
        )
        output_path = "plot/lr_test.png"
        (x, y), lr = lr_finder.plot(plot=False, output_path=output_path)
        plt.show(block=False)
        plt.pause(1.5)
        plt.close()
        # find if fig exist
        assert os.path.isfile(output_path)

        # check reset
        for (k1, v1), (k2, v2) in zip(
            model_state.items(), self.trainer.model.state_dict().items()
        ):
            assert k1 == k2
            assert (v1 == v2).all(), "parameters not reseted"

        for (k1, v1), (k2, v2) in zip(
            optimizer_state.items(),
            self.trainer.optimizer.state_dict().items(),
        ):
            assert k1 == k2
            assert v1 == v2, "parameters not reseted"

    def test_linear_scheduling_with_plot(self):
        model_state = copy.deepcopy(self.trainer.model.state_dict())
        optimizer_state = copy.deepcopy(self.trainer.optimizer.state_dict())

        lr_finder = LRFinder(self.trainer)
        #  lr_finder.fit(self.train_dl, self.val_dl, step_mode="linear")
        lr_finder.fit(self.train_dl, step_mode="linear")
        assert (
            self.handler.messages["info"][-1]
            == "Learning rate search has finished. Model and "
            "optimizer weights has been restored."
        )
        output_path = "plot/lr_test.png"
        (x, y), lr = lr_finder.plot(plot=False, output_path=output_path)
        plt.show(block=False)
        plt.pause(1.5)
        plt.close()
        # find if fig exist
        assert os.path.isfile(output_path)

        # check reset
        for (k1, v1), (k2, v2) in zip(
            model_state.items(), self.trainer.model.state_dict().items()
        ):
            assert k1 == k2
            assert (v1 == v2).all(), "parameters not reseted"

        for (k1, v1), (k2, v2) in zip(
            optimizer_state.items(),
            self.trainer.optimizer.state_dict().items(),
        ):
            assert k1 == k2
            assert v1 == v2, "parameters not reseted"
