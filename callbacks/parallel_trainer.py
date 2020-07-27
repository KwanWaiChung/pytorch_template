from . import Callback
import torch.nn as nn


class ParallelTrainer(Callback):
    def on_train_begin(self, logs):
        self.trainer.model = nn.DataParallel(self.trainer.model).to(
            self.trainer.device
        )

    def on_train_end(self, logs):
        self.trainer.model = self.trainer.model.module
