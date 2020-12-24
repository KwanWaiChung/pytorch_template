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

    def on_test_batch_end(self, logs):
        """
        Perhaps it's more direct to log customly in trainer, or output to
        a csv file.
        """
        #  if "table" in logs:
        #      columns = list(logs["table"][0].keys())
        #      data = [
        #          [example[key] for key in columns] for example in logs["table"]
        #      ]
        #      table = wandb.Table(data=data, columns=columns)
        #      wandb.log({"test examples": table})
        return

    def on_train_end(self, logs):
        if best_filepath := self.trainer.get_best_filepath():
            wandb.save(best_filepath)
