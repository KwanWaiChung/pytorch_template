from ....data import Field, Iterator, TabularDataset
from ....utils import spacy_tokenize, TextEncoder
from ..stub import DummyIterator
import torch
import random
import os

# 170 -> 101010, good balance of 1 and 0
torch.manual_seed(170)
random.seed(170)


def _postprocessor(example):
    example.stars = 1 if example.stars > 3 else 0
    return example


def getYelpDataloader(full=True):
    """Example Yelp dataset

        batch size: 4
        max sequence length: 50
        label: `1` if `stars` > 3, 0 otherwise.

    Args:
        full (bool): If True, use all examples.
            Otherwise, use about 100 examples.
    """
    text = Field(
        tokenizer=spacy_tokenize,
        is_sequential=True,
        to_lower=True,
        fix_length=50,
        batch_first=True,
    )
    label = Field(
        is_sequential=False,
        is_target=True,
        to_lower=False,
        to_numericalize=False,
    )

    ds = TabularDataset(
        path=getJsonFilename(),
        format="json",
        fields={"text": text, "stars": label},
        reader_params={"encoding": "utf-8"},
        postprocessor=_postprocessor,
    )
    if not full:
        ds, _ = ds.split(split_ratio=0.1, stratify_field="stars")

    train_ds, val_ds = ds.split(split_ratio=0.8, stratify_field="stars")
    train_dl = Iterator(train_ds, 4)
    val_dl = Iterator(val_ds, 4)

    encoder = text.build_vocab(train_ds)
    return train_dl, val_dl, len(encoder)


def getLinearDataloader(input_size):
    train_X = torch.rand(80, input_size)
    train_y = (100 * train_X).sum(dim=1, keepdim=True) + 1

    val_X = torch.rand(80, input_size)
    val_y = (100 * val_X).sum(dim=1, keepdim=True) + 1 + torch.randn(80, 1)
    train_dl = DummyIterator(train_X, train_y)
    val_dl = DummyIterator(val_X, val_y)
    return train_dl, val_dl


def getTextData(n=50):
    """An example text file with no label

    Returns:
        List of str.
    """
    with open(
        f"{os.path.dirname(os.path.realpath(__file__))}"
        "/../../data/test_data.txt",
        "r",
        encoding="utf-8",
    ) as f:
        data = random.choices([l.strip() for l in f], k=n)
    return data


def getCsvFilename():
    return (
        f"{os.path.dirname(os.path.realpath(__file__))}/../../data/train.csv"
    )


def getJsonFilename():
    return (
        f"{os.path.dirname(os.path.realpath(__file__))}"
        "/../../data/reviews.json"
    )
