from . import Callback, ModelCheckpoint
import wandb
import os


class Wandblogger(Callback):
    """log metrics to wandb."""

    def __init__(
        self,
        project,
        name,
        save_code=True,
        config={},
    ):
        wandb.init(
            project=project,
            name=name,
            save_code=save_code,
            config=config,
            resume=True,
        )
        super().__init__()

    def on_train_begin(self, logs):
        wandb.watch(self.trainer.model, log="all", log_freq=1)

    def on_epoch_end(self, logs):
        # training metrics
        for metric in self.trainer.metrics:
            wandb.log({metric.name: logs[metric.name]}, step=logs["epoch_idx"])
        wandb.log({"loss": logs["loss"]}, step=logs["epoch_idx"])

        for k, v in logs.items():
            if k.startswith("val_"):
                wandb.log({k: v}, step=logs["epoch_idx"])
        return False

    def on_train_end(self, logs):
        for v in self.trainer.callbacks.callbacks:
            if isinstance(v, ModelCheckpoint):
                wandb.save(v.best_filepath)
