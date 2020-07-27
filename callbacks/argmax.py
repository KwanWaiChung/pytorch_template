from . import Callback


class Argmax(Callback):
    def on_loss_end(self, logs):
        logs["last_y_pred"] = logs["last_y_pred"].argmax(dim=1)
