import torch.utils.data
import pandas as pd
import json
import os
import pickle
from multiprocessing.pool import Pool
from time import time
from typing import Union, List, Callable, Dict, Any, Tuple
from .example import Example
from ..utils import get_split_ratio, getlogger
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
        examples: List["Example"],
        fields: Dict[str, "Field"],
        filter_pred: Callable[["Example"], bool] = None,
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
            sort_key (Callable[[Example]]): A key to use for sorting examples.
                Usually its the length of some attributes.
            n_jobs (int): The number of jobs to use for the computation.
                -1 means using all processors. Default: -1.

        """
        self.logger = getlogger(__name__)
        if n_jobs == -1:
            p = Pool()
        else:
            p = Pool(n_jobs)

        self.fields = fields
        self.sort_key = sort_key
        self.examples = []
        if examples:
            self.logger.info(
                "Starting to preprocess the %d examples", len(examples)
            )

            current_time = time()
            self.examples = p.map(
                partial(_preprocess, fields=fields), examples
            )
            self.logger.info(
                "Finished preprocessing the examples. Total time: %f",
                time() - current_time,
            )

        if filter_pred:
            self.examples = [
                example for example in self.examples if filter_pred(example)
            ]

    @classmethod
    def fromProcessedExamples(
        cls,
        examples: List["Example"],
        fields: Dict[str, "Field"],
        sort_key: Callable[["Example"], int] = None,
    ):
        """
        Args:
            examples (List[Example]): A list holding all the preprocessed
                `Example` object.
            fields (Dict[str, Feild]): A dict that maps the name of
                each attribute/columns of the examples to a Field, which
                specified how to process them afterwards.
            sort_key (Callable[[Example]]): A key to use for sorting examples.
                Usually its the length of some attributes.

        Returns:
            Dataset object.

        """
        ds = Dataset(examples=[], fields=fields, sort_key=sort_key)
        ds.examples = examples
        return ds

    def split(
        self,
        split_ratio: Union[List[float], float, int],
        stratify_field: str = None,
        seed: int = 0,
    ) -> Tuple["Dataset", "Dataset"]:
        """Create train-test(-valid) splits from the examples.

        Args:
            split_ratio (Union[List[float], float, int]): if float, should be
                between 0.0 and 1.0 and represent the proportion of the dataset
                to include in the train split.
                If List[float], should be of length of 2 or 3 indicating the
                portion for train, test and (val).
                If int, represents the absolute number of train samples.
            stratify_field (bool): The `Field` name of the examples to
                stratify.
            seed (int): The seed for splitting.

        Returns:
            Tuple(Dataset, Dataset): The tuple of train, test, val
                (if len(split_ratio) == 3) `Dataset` objects.

        Raises:
            ValueError: if stratify_field not in self.fields.

        Examples:
            >>> ds = Dataset(examples, fields)
            >>> train_ds, test_ds, val_ds = ds.split(
                    [0.8, 0.1, 0.1], stratify_field="label"
                )
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

        if n_val > 0:
            train_examples, val_examples = train_test_split(
                train_examples,
                test_size=n_val,
                stratify=[
                    getattr(example, stratify_field)
                    for example in train_examples
                ]
                if stratify_field
                else None,
                random_state=seed,
            )
            val_dataset = Dataset.fromProcessedExamples(
                val_examples, self.fields, self.sort_key
            )
            train_dataset = Dataset.fromProcessedExamples(
                train_examples, self.fields, self.sort_key
            )
            test_dataset = Dataset.fromProcessedExamples(
                test_examples, self.fields, self.sort_key
            )
            return train_dataset, test_dataset, val_dataset

        train_dataset = Dataset.fromProcessedExamples(
            train_examples, self.fields, self.sort_key
        )
        test_dataset = Dataset.fromProcessedExamples(
            test_examples, self.fields, self.sort_key
        )
        return train_dataset, test_dataset

    def dump(self, filename: str):
        """
        Store the dataset object to avoid preprocessing again.

        Args:
            filename: The filename for the binary pickle file.

        Examples:
            >>> ds = Dataset(examples, fields)
            ds.dump("ds.pickle")
            ds = Dataset.load("ds.pickle")
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str):
        with open(filename, "rb") as f:
            encoder = pickle.load(f)
        return encoder

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

    def __getstate__(self):
        return {
            "examples": self.examples,
            "fields": self.fields,
            "sort_key": self.sort_key,
        }

    def __setstate__(self, d):
        self.examples = d["examples"]
        self.fields = d["fields"]
        self.sort_key = d["sort_key"]


class TabularDataset(Dataset):
    """Defines a Dataset of columns stored in CSV or JSON format."""

    def __init__(
        self,
        path: str,
        format: str,
        fields: Dict[str, "Field"],
        reader_params: Dict[str, Any],
        postprocessor: Callable[[List["Example"]], List["Example"]] = None,
        filter_pred: Callable[[Example], bool] = None,
        sort_key: Callable[["Example"], int] = None,
        n_jobs: int = -1,
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
            postprocessor (Callable): Apply to the list of Example obtained
                after reading the file from pandas.read_csv() or
                json.load().
            filter_pred (Callable[[Example], bool]): A predicate function
                that filters out the examples. Only the examples of which
                the predicate evalutes to `True` will be used. Default is None.
            sort_key (Callable[[Example]]): A key to use for sorting examples.
                Usually its the length of some attributes.
            n_jobs (int): The number of jobs to use for the preprocessing.
                -1 means using all processors. Default: -1.

        Raises:
            ValueError: If `path` doesn't exist.
            ValueError: If `format` is not 'csv' or 'json'.

        Examples:
            >>> text = Field(
                    is_sequential=True,
                    to_lower=True,
                    fix_length=150,
                    batch_first=True,
                )
                label = Field(is_sequential=False, is_target=True)
                ds = TabularDataset(
                    path="train.csv",
                    format="csv",
                    fields={"comment_text": text, "toxic": label},
                    reader_params={"encoding": "utf-8"},
                )
        """
        if not os.path.isfile(path):
            raise ValueError(f"The path '{path}' doesn't exist.")
        format = format.lower()
        if format not in ["csv", "json"]:
            raise ValueError(
                f"Format must be one of 'csv' or 'json', received '{format}'"
            )
        reader = {"csv": self.read_csv, "json": self.read_json}[format]
        examples = reader(path, reader_params, fields, postprocessor)
        super().__init__(examples, fields, filter_pred, sort_key, n_jobs)

    @staticmethod
    def read_csv(
        path: str,
        params: Dict[str, Any],
        fields: Dict[str, "Field"],
        postprocessor: Callable[
            [List[Dict[str, Any]]], List[Dict[str, Any]]
        ] = None,
    ) -> List["Example"]:
        """Read csv and return list of dict of attributes to values

        Args:
            path (str): Path of the csv file.
            params (Dict[str, Any]): Parameters for pd.read_csv method.
            postprocessor (Callable): postprocess the list of examples.

        """
        df = pd.read_csv(path, **params)
        df = df.to_dict("records")
        df = [Example.fromdict(d, fields) for d in df]
        if postprocessor:
            df = [postprocessor(example) for example in df]
        return df

    @staticmethod
    def read_json(
        path: str,
        params: Dict[str, Any],
        fields: Dict[str, "Field"],
        postprocessor: Callable[[Any], List[Dict]] = None,
    ) -> List["Example"]:
        """Read json and return list of dict of attributes to values

        Args:
            path (str): Path of the json file.
            params (Dict[str, Any]): Parms for the open() function.
            postprocessor (Callable): postprocess the list of examples.

        """
        with open(path, "r", **params) as f:
            j = json.load(f)
        j = [Example.fromdict(d, fields) for d in j]
        if postprocessor:
            j = [postprocessor(example) for example in j]
        return j
