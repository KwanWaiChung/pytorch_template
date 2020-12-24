from ..callbacks import Callback


class Metric(Callback):
    def __init__(self, name: str):
        self.name = name
        super().__init__()

    def __call__(self, y_true, y_pred):
        """
        Args:
            y_true (torch.tensor): Ground truth (correct) target values.
            y_pred (torch.tensor): Estimated targets returned by a classifier.

        Return:
            float: The average metric of all seen examples.
        """
        raise NotImplementedError(
            "Custom Metrics must implement this function"
        )

    def reset(self):
        raise NotImplementedError(
            "Custom Metrics must implement this function"
        )

    def on_epoch_begin(self, logs):
        self.reset()

    def on_train_batch_end(self, logs):
        logs[self.name] = self(logs["y_true"], logs["y_pred"])

    def on_val_begin(self, logs):
        self.reset()

    def on_val_batch_end(self, logs):
        logs["val_" + self.name] = self(logs["y_true"], logs["y_pred"])

    def on_test_begin(self, logs):
        self.reset()

    def on_test_batch_end(self, logs):
        if "y_true" in logs:
            logs["test_" + self.name] = self(logs["y_true"], logs["y_pred"])


class Accuracy(Metric):
    def __init__(self, pad_id=None):
        self.n_samples = 0
        self.n_correct = 0
        self.pad_id = pad_id
        super().__init__("acc")

    def __call__(self, y_true, y_pred):
        self.n_samples += y_true.shape[0]
        self.n_correct += (y_true == y_pred).sum().item()
        return self.n_correct / self.n_samples

    def reset(self):
        self.n_samples = 0
        self.n_correct = 0
