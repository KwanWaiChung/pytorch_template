from . import Callback
from torch.utils.tensorboard import SummaryWriter


class TensorBoard(Callback):
    """log metrics into a directory for visualization within the TensorBoard UI
    """

    def __init__(self, log_dir=None):
        self.writer = SummaryWriter(log_dir)
        super().__init__()

    def on_epoch_end(self, logs=None):
        # training metrics
        log_data = {
            metric.name + "/train": logs[metric.name]
            for metric in self.trainer.metrics
        }
        log_data["loss/train"] = logs["loss"]

        # validation metrics
        if self.trainer.use_val:
            log_data.update(
                {
                    k[4:] + "/val": v
                    for k, v in logs.items()
                    if k.startswith("val_")
                }
            )
            if "val_loss" in logs:
                log_data["loss/val"] = logs["val_loss"]

        for k, v in log_data.items():
            self.writer.add_scalar(k, v, logs["epoch"])
        return False
