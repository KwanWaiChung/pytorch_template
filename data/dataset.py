import torch.utils.data
from multiprocessing.pool import Pool
from typing import Union, List, Callable
from ..utils.misc import get_split_ratio
from .field import Field


class Dataset(torch.utils.data.Dataset):
    """Defines a dataset which contains `X` and `y`
        after preprocessed and tokenized (if required)
    """

    def __init__(
        self,
        examples: List[List[str, Union[str, int]]],
        fields: List[Field],
        filter_pred: Callable[List[str, Union[str, int]], bool] = None,
        n_jobs: int = -1,
    ):
        """
        Args:
            examples (List[List[str, Union[str, int]]]): A list holding
                all the raw examples each example would be a list containing
                the `x` (str) and the `y` (str/int). Both x and y are just
                raw strings without preprocessing
            fields (List[field]): A list holding the corresponding `Field`
                object for `X` and `y` respectively. The `Field` object
                specifies how to process with them.
            filter_pred (Callable[Tuple[str, Union[str, int]], bool]): A
                predicate function that filters out the examples. Only
                the examples of which the predicate evalutes to `True`
                will be used. Default is None.
            n_jobs (int): The number of jobs to use for the computation.
                -1 means using all processors. Default: -1.
        """
        # TODO: Add test case (preprocessing)
        if n_jobs == -1:
            p_X = Pool()
            p_y = Pool()
        else:
            p_X = Pool(n_jobs)
            p_y = Pool(n_jobs)

        self.X = p_X.map_async(
            fields[0].preprocess, (example[0] for example in examples)
        )
        self.y = p_y.map_async(
            fields[1].preprocess, (example[1] for example in examples)
        )

        # TODO: Add test case (filtering examples)
        if filter_pred:
            examples = [
                example
                for example in zip(self.X, self.y)
                if filter_pred(example)
            ]
            self.X, self.y = [list(dummy) for dummy in zip(*examples)]

    def split(
        self,
        split_ratio: Union[List[float], float],
        stratified=False,
        strat_field=None,
        random_state=None,
    ):
        """Create train-test(-valid) splits from the examples"""
        train_ratio, test_ratio, val_ratio = get_split_ratio(split_ratio)
