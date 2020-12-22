from .logger import getlogger
from typing import Union, List, Tuple
from allennlp.data import AllennlpDataset
from sklearn.model_selection import train_test_split
import numpy as np
import random
import torch

logger = getlogger()


def get_split_ratio(split_ratio: Union[List[float], float]):
    """Get the train, test, val ratio and check that the split is correct"""
    valid_ratio = 0.0
    if isinstance(split_ratio, float):
        assert (
            0.0 < split_ratio < 1.0
        ), f"Split ratio f{split_ratio} is not between 0 and 1"
        test_ratio = round(1.0 - split_ratio, 10)
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


def split(
    dataset: AllennlpDataset,
    split_ratio: Union[List[float], float, int],
    stratify: List,
    seed: int = 0,
) -> Tuple[AllennlpDataset, AllennlpDataset]:
    train_ratio, test_ratio, val_ratio = get_split_ratio(split_ratio)
    n_test = int(test_ratio * len(dataset))
    n_val = int(val_ratio * len(dataset))

    train_instances, test_instances = train_test_split(
        dataset.instances,
        test_size=n_test,
        stratify=stratify,
        random_state=seed,
    )
    if n_val > 0:
        train_instances, val_instances = train_test_split(
            train_instances,
            test_size=n_val,
            stratify=stratify,
            random_state=seed,
        )
        return (
            AllennlpDataset(train_instances),
            AllennlpDataset(test_instances),
            AllennlpDataset(val_instances),
        )
    return AllennlpDataset(train_instances), AllennlpDataset(test_instances)


def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_deterministic(True)
