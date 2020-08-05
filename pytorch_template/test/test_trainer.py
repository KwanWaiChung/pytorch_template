import torch
import torch.nn as nn
from ..trainer import BaseTrainer
from ..metrics import Accuracy
from ..callbacks import Argmax
from .utils.model import LSTM, LinearRegression
from .utils.dataset import getLinearDataloader, getYelpDataloader

input_size = 1
output_size = 1
# 170 -> 101010, good balance of 1 and 0
torch.manual_seed(170)


def test_trainer_train_regression():
    """Check if trainer is coded correctly and coverges on simple
        regression data.
    """
    epoch = 20
    lr = 1e1
    train_dl, val_dl = getLinearDataloader(input_size)
    model = LinearRegression(input_size, output_size)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = BaseTrainer(model, criterion, optimizer)
    trainer.fit(train_dl, epoch, val_dl)

    for i in range(1, epoch):
        assert (
            trainer.history.history["loss"][i - 1]
            > trainer.history.history["loss"][i]
        ) or (
            trainer.history.history["loss"][i - 2]
            > trainer.history.history["loss"][i]
        )


#  Yelp metrics for seed 170
#  acc=0.5000, loss=0.716, val_loss=0.6856, val_acc=0.4706
#  acc=0.6429, loss=0.638, val_loss=0.5944, val_acc=0.7059
#  acc=1.0000, loss=0.541, val_loss=0.4788, val_acc=1.0000
#  acc=1.0000, loss=0.361, val_loss=0.3656, val_acc=1.0000
#  acc=1.0000, loss=0.154, val_loss=0.3423, val_acc=1.0000
#  acc=1.0000, loss=0.0504, val_loss=0.7797, val_acc=0.8824
#  acc=1.0000, loss=0.0188, val_loss=1.6339, val_acc=0.8824
#  acc=1.0000, loss=0.0091, val_loss=1.3140, val_acc=0.9412
#  acc=1.0000, loss=0.00539, val_loss=1.5644, val_acc=0.9412
#  acc=1.0000, loss=0.00364, val_loss=3.3879, val_acc=0.8824
def test_trainer_train_yelp_classification():
    """Check if trainer converges on text classification.
        Consider it closest to real world task.
    """
    lr = 1e-2
    train_dl, val_dl, vocab_size = getYelpDataloader(full=False)

    model = LSTM(
        vocab_size=vocab_size,
        embedding_dim=30,
        hidden_dim=10,
        n_layers=1,
        n_classes=2,
    )
    trainer = BaseTrainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        metrics=[Accuracy()],
        callbacks=[Argmax()],
    )
    trainer.fit(train_dl=train_dl, n_epochs=10, val_dl=val_dl)
