from .logger import getlogger
from typing import Union, List
import numpy as np

logger = getlogger()


def get_split_ratio(split_ratio: Union[List[float], float]):
    """Get the train, test, val ratio and check that the split is correct"""
    valid_ratio = 0.0
    if isinstance(split_ratio, float):
        assert (
            0.0 < split_ratio < 1.0
        ), f"Split ratio f{split_ratio} is not between 0 and 1"
        test_ratio = 1.0 - split_ratio
        return split_ratio, test_ratio, valid_ratio
    elif isinstance(split_ratio, list):
        length = len(split_ratio)
        assert (
            length == 2 or length == 3
        ), f"Length of split ratio list should be 2 or 3, got {split_ratio}"

        # Normalize if necessary
        ratio_sum = sum(split_ratio)
        if not ratio_sum == 1.0:
            logger.info(
                "Trying to normalize the split ratio as %s doesn't sum to 1",
                split_ratio,
            )
            split_ratio = [ratio / ratio_sum for ratio in split_ratio]
        if length == 2:
            return tuple(split_ratio + [valid_ratio])
        return tuple(split_ratio)
    else:
        raise ValueError(
            f"Split ratio must be a float or a list, got {type(split_ratio)}"
        )


def pad(
    x: List[str],
    max_len: int = None,
    pad_token: str = "<pad>",
    sos_token: str = None,
    eos_token: str = None,
    pad_first: bool = False,
    truncate_first: bool = False,
):
    paddings = [pad_token] * max(0, max_len - len(x))
    sos_token = [sos_token] if sos_token else []
    eos_token = [eos_token] if eos_token else []
    truncated_tokens = x[-max_len:] if truncate_first else x[:max_len]
    if pad_first:
        return paddings + sos_token + truncated_tokens + eos_token
    else:
        return sos_token + truncated_tokens + eos_token + paddings


def flatten_2d(y):
    """Convert columns or 1d list (unchange) to 1d list"

    Args:
        y (list): The list to convert.

    Returns:
        y (list): The flattened 1d list.
    """
    warned = False
    res = []
    for ele in y:
        if isinstance(ele, list):
            if not warned:
                logger.info(
                    "A column-vector y was passed when a 1d array was"
                    " expected. Please change the shape of y to "
                    "(n_samples, ), for example using ravel()."
                    " The column-vector is converted to a 1d array"
                )
                warned = True
            for _ele in ele:
                if isinstance(_ele, list):
                    raise ValueError("Input is not 2d or 1d list")
            res += ele
        else:
            res.append(ele)
    return res


def batch(examples, batch_size):
    for i in range(0, len(examples), batch_size):
        minibatch = examples[i : i + batch_size]
        yield minibatch
