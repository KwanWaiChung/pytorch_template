import torch.utils.data
from multiprocessing.pool import Pool
from typing import Union, List, Callable
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
            filter_pred (Callable[List[str, Union[str, int]], bool]): A
                predicate function that filters out the examples. Only
                the examples of which the predicate evalutes to `True`
                will be used. Default is None.
            n_jobs (int): The number of jobs to use for the computation.
                -1 means using all processors. Default: -1.
        """
        # TODO: Preprocess first, (multi-processing)
        if n_jobs == -1:
            p_x = Pool()
            p_y = Pool()
        else:
            p_x = Pool(n_jobs)
            p_y = Pool(n_jobs)

        self.x = p_x.map_async(
            fields[0].preprocess, (example[0] for example in examples)
        )
        self.y = p_y.map_async(
            fields[1].preprocess, (example[1] for example in examples)
        )

        # TODO: Filter out the examples (if needed)

        if filter_pred:
            pass
