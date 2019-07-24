from .logger import getlogger
from typing import Union, List

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
