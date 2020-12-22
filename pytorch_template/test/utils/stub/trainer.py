from ....trainer import BaseTrainer
from time import sleep


class RegressionTrainer(BaseTrainer):
    def training_step(self, batch_dict, batch_idx):
        output = self.model(batch_dict["train_X"])
        loss = self.criterion(output, batch_dict["train_y"])
        return {
            "loss": loss,
            "batch_size": batch_dict["train_X"].shape[0],
            "y_pred": output,
            "y_true": batch_dict["train_y"],
        }

    def validation_step(self, batch_dict, batch_idx):
        output = self.model(batch_dict["train_X"])
        loss = self.criterion(output, batch_dict["train_y"])
        return {
            "val_loss": loss,
            "batch_size": batch_dict["train_X"].shape[0],
            "y_pred": output,
            "y_true": batch_dict["train_y"],
        }


class ClassificationTrainer(BaseTrainer):
    def __init__(self, delay=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay = delay

    def training_step(self, batch_dict, batch_idx):
        output = self.model(batch_dict["text"])
        loss = self.criterion(output, batch_dict["label"])
        return {
            "loss": loss,
            "batch_size": batch_dict["label"].shape[0],
            "y_pred": output.max(dim=-1)[1],
            "y_true": batch_dict["label"],
        }

    def validation_step(self, batch_dict, batch_idx):
        output = self.model(batch_dict["text"])
        loss = self.criterion(output, batch_dict["label"])
        return {
            "val_loss": loss,
            "batch_size": batch_dict["label"].shape[0],
            "y_pred": output.max(dim=-1)[1],
            "y_true": batch_dict["label"],
        }

    def fit_epoch(self, train_dl, val_dl):
        super().fit_epoch(train_dl, val_dl)
        if self.delay:
            sleep(1)
