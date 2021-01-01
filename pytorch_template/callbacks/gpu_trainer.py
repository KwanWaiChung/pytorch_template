from . import Callback
import torch


class GpuTrainer(Callback):
    def __init__(self, device_id: int = None):
        if device_id is None or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:" + str(device_id))

    def on_train_begin(self, logs):
        self.trainer.model.to(self.device)

    def on_train_batch_begin(self, logs):
        self._move_tensors_to_gpu(logs["batch_dict"])

    def on_val_batch_begin(self, logs):
        self._move_tensors_to_gpu(logs["batch_dict"])

    def on_test_batch_begin(self, logs):
        self._move_tensors_to_gpu(logs["batch_dict"])

    def _move_tensors_to_gpu(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                self._move_tensors_to_gpu(v)
            elif isinstance(v, torch.Tensor):
                d[k] = v.to(self.device)
