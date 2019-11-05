import torch.utils.data
from multiprocessing.pool import Pool
from time import time
from typing import Union, List, Callable, Dict
from ..utils.misc import get_split_ratio
from .example import Example
from ..utils.logger import getlogger
from functools import partial


def _preprocess(example, fields):
    return example.preprocess(fields)


class Dataset(torch.utils.data.Dataset):
    """Defines a dataset composed of Examples along with its Fields

    Attributes:
        examples (List[Example]): A list holding all the unprocessed
            examples. Each example will have attributes as indicated
            in the keys of `fields`.
        fields (Dict[str, Feild]): A dict that maps the name of
            each attribute/columns of the examples to a Field, which
            specified how to process them afterwards.
    """

    def __init__(
        self,
        examples: List[Example],
        fields: Dict[str, "Field"],
        filter_pred: Callable[[Example], bool] = None,
        sort_key: Callable[["Example"], int] = None,
        n_jobs: int = -1,
    ):
        """
        Args:
            examples (List[Example]): A list holding all the unprocessed
                examples. Each example will have attributes as indicated
                in the keys of `fields`.
            fields (Dict[str, Feild]): A dict that maps the name of
                each attribute/columns of the examples to a Field, which
                specified how to process them afterwards.
            filter_pred (Callable[[Example], bool]): A predicate function
                that filters out the examples. Only the examples of which
                the predicate evalutes to `True`will be used. Default is None.
            n_jobs (int): The number of jobs to use for the computation.
                -1 means using all processors. Default: -1.
        """
        if n_jobs == -1:
            p = Pool()
        else:
            p = Pool(n_jobs)

        self.logger = getlogger(__name__)

        self.logger.info(
            "Starting to preprocess the %d examples", len(examples)
        )

        current_time = time()
        self.examples = p.map(partial(_preprocess, fields=fields), examples)
        self.fields = fields
        self.sort_key = sort_key
        self.logger.info(
            "Finished preprocessing the examples. Total time: %f",
            time() - current_time,
        )

        if filter_pred:
            self.examples = [
                example for example in self.examples if filter_pred(example)
            ]

    def split(
        self,
        split_ratio: Union[List[float], float],
        stratified=False,
        strat_field=None,
        random_state=None,
    ):
        """Create train-test(-valid) splits from the examples"""
        train_ratio, test_ratio, val_ratio = get_split_ratio(split_ratio)

    def __getitem__(self, i):
        return self.examples[i]

    def __iter__(self):
        for x in self.examples:
            yield x

    def __len__(self):
        return len(self.examples)

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)
