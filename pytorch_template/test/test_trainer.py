import torch
import torch.nn as nn
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
from ..trainer import BaseTrainer
from ..metrics import Accuracy
from ..callbacks import Argmax
from .utils.model import LSTM, LinearRegression
from .utils.dataset import getLinearDataloader, getYelpDataloader
from .utils.stub import RegressionTrainer, ClassificationTrainer
from ..utils.misc import set_seed

input_size = 1
output_size = 1
# 170 -> 101010, good balance of 1 and 0
SEED = 170


def test_trainer_train_regression():
    """Check if trainer is coded correctly and coverges on simple
    regression data.
    """
    set_seed(170)
    epoch = 20
    lr = 1e-1
    train_dl, val_dl = getLinearDataloader(input_size)
    model = LinearRegression(input_size, output_size)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = RegressionTrainer(
        model,
        criterion,
        optimizer,
        early_stopping=None,
        model_checkpoint=None,
    )
    trainer.fit(train_dl, epoch, train_dl)

    for i in range(1, epoch):
        assert (
            trainer.history.history["loss"][i - 1]
            > trainer.history.history["loss"][i]
        ) or (
            trainer.history.history["loss"][i - 2]
            > trainer.history.history["loss"][i]
        )


#  Yelp metrics for seed 170
#  Epoch 1/20: acc=0.6250, loss=0.6803, val_loss=0.6704, val_acc=0.6333
#  Epoch 2/20: acc=0.7083, loss=0.5509, val_loss=0.6076, val_acc=0.7000
#  Epoch 3/20: acc=0.9167, loss=0.3965, val_loss=0.5325, val_acc=0.8667
#  Epoch 4/20: acc=0.9583, loss=0.2908, val_loss=0.2953, val_acc=0.9667
#  Epoch 5/20: acc=0.9583, loss=0.2121, val_loss=0.3009, val_acc=0.9333
#  Epoch 6/20: acc=0.9583, loss=0.1882, val_loss=0.6063, val_acc=0.9000
#  Epoch 7/20: acc=0.9583, loss=0.1714, val_loss=0.8819, val_acc=0.9000
#  Epoch 8/20: acc=0.9583, loss=0.1728, val_loss=1.1203, val_acc=0.9000
#  Epoch 9/20: acc=0.9583, loss=0.1631, val_loss=1.2840, val_acc=0.9000
#  Epoch 10/20: acc=0.9583, loss=0.1590, val_loss=1.4157, val_acc=0.9000
#  Epoch 11/20: acc=0.9583, loss=0.1586, val_loss=1.5130, val_acc=0.9000
#  Epoch 12/20: acc=0.9583, loss=0.1543, val_loss=1.5956, val_acc=0.9000
#  Epoch 13/20: acc=0.9583, loss=0.1545, val_loss=1.6668, val_acc=0.9000
#  Epoch 14/20: acc=0.9583, loss=0.1517, val_loss=1.7376, val_acc=0.9000
#  Epoch 15/20: acc=0.9583, loss=0.1484, val_loss=1.8136, val_acc=0.9000
#  Epoch 16/20: acc=0.9583, loss=0.1437, val_loss=1.8995, val_acc=0.9000
#  Epoch 17/20: acc=0.9583, loss=0.1185, val_loss=2.0064, val_acc=0.9000
#  Epoch 18/20: acc=0.9167, loss=0.5090, val_loss=1.7997, val_acc=0.8667
#  Epoch 19/20: acc=1.0000, loss=0.0323, val_loss=1.2229, val_acc=0.9333
#  Epoch 20/20: acc=1.0000, loss=0.0271, val_loss=0.6185, val_acc=0.9667
def test_trainer_train_yelp_classification():
    """Check if trainer converges on text classification.
    Consider it closest to real world task.
    """
    set_seed(170)
    lr = 1e-2
    train_dl, val_dl, vocab_size = getYelpDataloader()

    epoch = 20

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
        early_stopping=None,
        model_checkpoint=None,
        metrics=[Accuracy()],
    )
    trainer.fit(train_dl=train_dl, n_epochs=epoch, val_dl=val_dl)
    assert trainer.history.history["loss"][-1] < 0.1
    assert trainer.history.history["acc"][-1] == 1
