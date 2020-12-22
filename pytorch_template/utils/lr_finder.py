import torch
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
from typing import Iterator
from .logger import getlogger


class LRFinder(object):
    """Learning rate range test.

    The learning rate range test increases the learning rate in a pre-training
    run between two boundaries in a linear or exponential manner. It provides
    valuable information on how well the network can be trained over a range of
    learning rates and what is the optimal learning rate.

    Arguments:
        model (torch.nn.Module): Wrapped model.
        optimizer (torch.optim.Optimizer): Wrapped optimizer where the defined
            learning is assumed to be the lower boundary of the range test.
        criterion (torch.nn.Module): Wrapped loss function.
        memory_cache (bool, optional): If this flag is set to True,
            `state_dict` of model and optimizer will be cached in memory.
            Otherwise, they will be saved to files under the `cache_dir`.
            Default: True.
        cache_dir (str): Path for storing temporary files. If no path is
            specified, system-wide temporary directory is used.
            Notice that this parameter will be ignored if `memory_cache`
            is True.

    Example:
        >>> lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
        >>> lr_finder.range_test(dataloader, end_lr=100, num_iter=100)

    Cyclical Learning Rates for Training Neural Networks:
        https://arxiv.org/abs/1506.01186
    fastai/lr_find: https://github.com/fastai/fastai
    """

    def __init__(self, trainer: "BaseTrainer"):
        self.logger = getlogger(__name__)
        self.trainer = trainer
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.use_val = False

        # Save the original state of the model and optimizer so they
        # can be restored if needed
        self.memory = {
            "model": copy.deepcopy(self.trainer.model.state_dict()),
            "optimizer": copy.deepcopy(self.trainer.optimizer.state_dict()),
        }

    def _reset(self):
        """Restores the model and optimizer to their initial states."""
        self.trainer.model.load_state_dict(self.memory["model"])
        self.trainer.optimizer.load_state_dict(self.memory["optimizer"])

    def _forever_dataloader(self, dl):
        while True:
            for batch_dict in dl:
                yield batch_dict

    def fit(
        self,
        train_dl: Iterator,
        val_dl: Iterator = None,
        str_lr=1e-7,
        end_lr=10,
        num_iter=100,
        step_mode="exp",
        smooth_f=0.05,
        diverge_th=5,
    ):
        """Performs the learning rate range test.

        Args:
            train_loader (torch.utils.data.DataLoader):
                The training set data laoder.
            val_loader (torch.utils.data.DataLoader, optional):
                If `None` the range test will only use the training loss.
                When given a data loader,the model is evaluated after each
                iteration on that dataset and the evaluation loss is used.
                Note that in this mode the test takes significantly longer but
                generally produces more precise results. Default: None.
            end_lr (float, optional): The maximum learning rate to test.
                Default: 10.
            num_iter (int, optional): The number of iterations over which the
                test occurs. Default: 100.
            step_mode (str, optional): One of the available learning rate
                policies, linear or exponential ("linear", "exp").
                Default: "exp".
            smooth_f (float, optional): The loss smoothing factor within the
                [0, 1] interval. Disabled if set to 0, otherwise the loss is
                smoothed using exponential smoothing. Default: 0.05.
            diverge_th (int, optional): The test is stopped when the loss
                surpasses the threshold(diverge_th * best_loss). Default: 5.
        """
        self._set_learning_rate(str_lr)
        if val_dl is not None:
            self.use_val = True
        # Reset test results
        self.history = {"lr": [], "loss": []}
        best_loss = float("inf")

        # Initialize the proper learning rate policy
        if step_mode.lower() == "exp":
            lr_schedule = ExponentialLR(
                self.trainer.optimizer, end_lr, num_iter
            )
        elif step_mode.lower() == "linear":
            lr_schedule = LinearLR(self.trainer.optimizer, end_lr, num_iter)
        else:
            raise ValueError(
                "expected one of (exp, linear), got {}".format(step_mode)
            )

        if smooth_f < 0 or smooth_f > 1:
            raise ValueError("smooth_f is outside the range [0, 1]")

        for i, batch_dict in enumerate(self._forever_dataloader(train_dl)):
            if i == num_iter:
                break

            self.history["lr"].append(lr_schedule.get_last_lr()[0])
            loss = self.trainer._training_step(batch_dict, i)["loss"]
            loss.backward()
            self.trainer.optimizer.step()

            if val_dl is not None:
                self.trainer._validate(val_dl)
                loss = torch.tensor(self.trainer.logs["val_loss"])
            lr_schedule.step()
            if i == 0:
                best_loss = loss
            else:
                if smooth_f > 0:
                    loss = (
                        smooth_f * loss
                        + (1 - smooth_f) * self.history["loss"][-1]
                    )
                if loss < best_loss:
                    best_loss = loss

            self.history["loss"].append(loss.item())
            if loss > diverge_th * best_loss:
                self.logger.info("Stopping early, the loss has diverged")
                break

        self._reset()
        self.logger.info(
            "Learning rate search has finished. Model and "
            "optimizer weights has been restored."
        )

    def _set_learning_rate(self, new_lr):
        for param_group in self.trainer.optimizer.param_groups:
            param_group["lr"] = new_lr

    def plot(
        self,
        skip_start=0,
        skip_end=0,
        log_lr=True,
        recommend=True,
        plot=False,
        output_path=None,
    ):
        """Plots the learning rate range test.

        Arguments:
            skip_start (int, optional): Number of batches to trim
                from the start. Default: 0.
            skip_end (int, optional): Number of batches to trim
                from the end. Default: 0.
            log_lr (bool, optional): True to plot the learning rate
                in a logarithmic scale; otherwise, plotted in a linear scale.
                Default: True.
        """

        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")

        # Get the data to plot from the history dictionary.
        # Also, handle skip_end=0 properly so the behaviour is the expected
        lrs = self.history["lr"]
        losses = self.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Plot loss as a function of the learning rate
        fig, ax = plt.subplots()
        ax.plot(lrs, losses)
        #  plt.plot(lrs, losses)
        if log_lr:
            #  plt.xscale("log")
            ax.set_xscale("log")
        #  plt.xlabel("Learning rate")
        #  plt.ylabel("Validation Loss" if self.use_val else "Training Loss")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Validation Loss" if self.use_val else "Training Loss")

        if recommend:
            min_grad_idx = (np.gradient(np.array(losses))).argmin()
            ax.scatter(
                lrs[min_grad_idx],
                losses[min_grad_idx],
                s=75,
                marker="o",
                color="red",
                zorder=2,
                label=f"steepest gradient({lrs[min_grad_idx]:.4f})",
            )
            ax.legend()
        if plot:
            plt.show()
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
        x = ax.lines[0].get_xdata()
        y = ax.lines[0].get_ydata()
        return (x, y), lrs[min_grad_idx] if recommend else (x, y)


class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        end_lr (float, optional): The initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): The number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): The index of last epoch. Default: -1.

    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [
            base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs
        ]


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a
        number of iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        end_lr (float, optional): The initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): The number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): The index of last epoch. Default: -1.

    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [
            base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs
        ]
