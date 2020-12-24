from ..metrics import Accuracy
from .utils.dataset import getYelpDataloader
from .utils.stub import ClassificationTrainer
from .utils.model import LSTM
from ..utils.misc import set_seed
import torch
import torch.nn as nn


def test_accuracy():
    acc = Accuracy()
    y_pred = torch.tensor([0, 2, 1, 3, 2])
    y_true = torch.tensor([0, 1, 2, 3, 0])
    logs = {"y_pred": y_pred, "y_true": y_true}
    acc.on_epoch_begin({})
    acc.on_train_batch_end(logs)
    assert acc.n_samples == 5
    assert acc.n_correct == 2
    assert logs["acc"] == 0.4

    # test accumulate train metrics
    logs = {"y_pred": torch.tensor([0, 1]), "y_true": torch.tensor([1, 1])}
    acc.on_train_batch_end(logs)
    assert acc.n_samples == 7
    assert acc.n_correct == 3
    # 3 / 7
    assert abs(logs["acc"] - 0.429) < 1e-3

    # test epoch reset
    acc.on_epoch_begin({})
    assert acc.n_samples == 0
    assert acc.n_correct == 0
    logs = {"y_pred": torch.tensor([0, 1]), "y_true": torch.tensor([1, 1])}
    acc.on_train_batch_end(logs)
    # 1/2
    assert acc.n_samples == 2
    assert acc.n_correct == 1
    assert logs["acc"] == 0.5

    # test val reset
    acc.on_val_begin({})
    assert acc.n_samples == 0
    assert acc.n_correct == 0
    logs = {"y_pred": torch.tensor([0, 1]), "y_true": torch.tensor([1, 1])}
    acc.on_val_batch_end(logs)
    # 1/2
    assert acc.n_samples == 2
    assert acc.n_correct == 1
    assert logs["val_acc"] == 0.5

    # test test reset
    acc.on_test_begin({})
    assert acc.n_samples == 0
    assert acc.n_correct == 0
    logs = {"y_pred": torch.tensor([0, 1]), "y_true": torch.tensor([1, 1])}
    acc.on_test_batch_end(logs)
    # 1/2
    assert acc.n_samples == 2
    assert acc.n_correct == 1
    assert logs["test_acc"] == 0.5

    acc.on_test_begin({})
    assert acc.n_samples == 0
    assert acc.n_correct == 0
    logs = {"y_pred": torch.tensor([0, 1])}
    acc.on_test_batch_end(logs)
    # no output
    assert acc.n_samples == 0
    assert acc.n_correct == 0
    assert "test_acc" not in logs


def test_accuracy_with_yelp():
    set_seed(1337)
    _, val_dl1, vocab_size = getYelpDataloader(val_batch_size=2)
    set_seed(1337)
    _, val_dl2, _ = getYelpDataloader(val_batch_size=-1)
    _, val_dl3, _ = getYelpDataloader(val_batch_size=1)
    lr = 1e-2

    model = LSTM(
        vocab_size=vocab_size,
        embedding_dim=30,
        hidden_dim=10,
        n_layers=1,
        n_classes=2,
    )
    trainer1 = ClassificationTrainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        early_stopping=None,
        model_checkpoint=None,
        metrics=[Accuracy()],
    )
    trainer2 = ClassificationTrainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        early_stopping=None,
        model_checkpoint=None,
        metrics=[Accuracy()],
    )
    trainer3 = ClassificationTrainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        early_stopping=None,
        model_checkpoint=None,
        metrics=[Accuracy()],
    )
    trainer1._validate(val_dl1)
    trainer2._validate(val_dl2)
    trainer3._validate(val_dl3)
    assert (
        trainer1.logs["val_acc"]
        == trainer2.logs["val_acc"]
        == trainer3.logs["val_acc"]
    )
