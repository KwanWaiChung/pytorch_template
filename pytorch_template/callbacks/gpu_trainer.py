from . import Callback
import torch


class GpuTrainer(Callback):
    def __init__(self, device_id=0):
        if device_id == "cpu" or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:" + str(device_id))

    def on_train_begin(self, logs):
        self.trainer.model.to(self.device)

    def on_train_batch_begin(self, logs):
        batch_dict = logs["batch_dict"]
        for k, v in batch_dict.items():
            if isinstance(v, torch.Tensor):
                batch_dict[k] = v.to(self.device)
        logs["batch_dict"] = batch_dict

    def on_val_batch_begin(self, logs):
        batch_dict = logs["batch_dict"]
        for k, v in batch_dict.items():
            if isinstance(v, torch.Tensor):
                batch_dict[k] = v.to(self.device)
        logs["batch_dict"] = batch_dict

    def on_test_batch_begin(self, logs):
        batch_dict = logs["batch_dict"]
        for k, v in batch_dict.items():
            if isinstance(v, torch.Tensor):
                batch_dict[k] = v.to(self.device)
        logs["batch_dict"] = batch_dict
