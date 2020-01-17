import torch.utils.data
import random
import pandas as pd
import json
import os
from multiprocessing.pool import Pool
from time import time
from typing import Union, List, Callable, Dict, Any
from ..utils.misc import get_split_ratio
from .example import Example
from ..utils.logger import getlogger
from functools import partial
from sklearn.model_selection import train_test_split


def _preprocess(example, fields):
    return example.preprocess(fields)


class Dataset(torch.utils.data.Dataset):
    """Defines a dataset composed of Examples along with its Fields

    It is iterable. We will loop over the Examples.
        >>> ds = Dataset(examples, dict(fields))
        >>> for example in ds:
                print(example.text)

    It is indexable. We can access the ith Example object.
        >>> ds = Dataset(examples, dict(fields))
        >>> example = ds[4]

    We can also loop over certain field of all Examples too.
        >>> ds = Dataset(examples, dict(fields))
        >>> for example in ds.text:
                print(example)

    Attributes:
        examples (List[Example]): A list holding all the preprocessed
            examples. Each example will have attributes as indicated
            in the keys of the attribute `fields`.
        fields (Dict[str, Field]): A dict that maps the name of
            each attribute/columns of the examples to a Field, which
            specified how to process them afterwards.
        sort_key (Callable[[Example]]): A key to use for sorting examples.
            Usually its the length of some attributes.
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
        split_ratio: Union[List[float], float, int],
        stratify_field: str = None,
        seed: int = 0,
    ):
        """Create train-test(-valid) splits from the examples

        Args:
            split_ratio (Union[List[float], float, int]): if float, should be
                between 0.0 and 1.0 and represent the proportion of the dataset
                to include in the train split.
                If List[float], should be of length of 2 or 3 indicating the
                portion for train, test and (val).
                If int, represents the absolute number of train samples.
            stratify_field (bool): The name of the examples `Field` to
                stratify.
            seed: The seed for splitting.
        """
        if stratify_field and stratify_field not in self.fields:
            raise ValueError(
                f"Invalid field name for stratify_field: {stratify_field}"
            )

        train_ratio, test_ratio, val_ratio = get_split_ratio(split_ratio)
        n_test = int(test_ratio * len(self))
        n_val = int(val_ratio * len(self))

        train_examples, test_examples = train_test_split(
            self.examples,
            test_size=n_test,
            stratify=list(getattr(self, stratify_field))
            if stratify_field
            else None,
            random_state=seed,
        )
        train_examples, val_examples = train_test_split(
            train_examples,
            test_size=n_val,
            stratify=[
                getattr(example, stratify_field) for example in train_examples
            ]
            if stratify_field
            else None,
            random_state=seed,
        )
        return train_examples, test_examples, val_examples

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


class TabularDataset(Dataset):
    """Defines a Dataset of columns stored in CSV or JSON format."""

    def __init__(
        self,
        path: str,
        format: str,
        fields: Dict[str, "Field"],
        reader_params: Dict[str, Any],
        postprocessor: Callable[[Any], List[Dict]] = None,
    ):
        """Create a Dataset given the path, file format, and fields.

        Args:
            path (str): Path to the data file.
            format (str): The format of the data file. One of "csv"
                or "json".
            fields (Dict[str, Field]): A dict that maps the columns
                of the data file to a Field. The keys must be a
                subset of the json keys or csv columns.
            reader_params (Dict[str, Any]): The extra parameters that got
                passed to pandas.read_csv() if csv or open() if json
            postprocessor (Callable): Apply to convert the format to List
                of Dict after reading the file from pandas.read_csv() or
                json.load().
        """
        if not os.path.isfile(path):
            raise ValueError(f"The path '{path}' doesn't exist.")
        format = format.lower()
        if format not in ["csv", "json"]:
            raise ValueError(
                f"Format must be one of 'csv' or 'json', received '{format}'"
            )
        reader = {"csv": self.read_csv, "json": self.read_json}[format]
        examples = reader(path, reader_params, postprocessor)

        @classmethod
        def read_csv(
            path: str,
            params: Dict[str, Any],
            postprocessor: Callable[[Any], List[Dict]] = None,
        ) -> List[Dict[str, Any]]:
            """Read csv and return list of dict of columns"""
            df = pd.read_csv(path, **params)
            if postprocessor:
                df = postprocessor(df)
            return df.to_dict("records")

        @classmethod
        def read_json(
            path: str,
            params: Dict[str, Any],
            postprocessor: Callable[[Any], List[Dict]] = None,
        ) -> List[Dict[str, Any]]:
            """Read json and return list of dict of columns"""
            with open(path, **params) as f:
                j = json.load(f)
            if postprocessor:
                j = postprocessor(j)
            return j
