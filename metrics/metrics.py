from ..callbacks.callbacks import Callback


class Metric(Callback):
    def __init__(self, name: str):
        self.name = name
        super().__init__()

    def __call__(self, y_true, y_pred):
        """
        Args:
            y_true (torch.tensor): Ground truth (correct) target values.
            y_pred (torch.tensor): Estimated targets returned by a classifier.
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
        logs[self.name] = self(logs["last_y_true"], logs["last_y_pred"])
