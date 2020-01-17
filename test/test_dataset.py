from time import sleep
from random import random
from ..data.field import Field
from ..data.dataset import Dataset
from ..data.example import Example
import os
import spacy

spacy = spacy.load("en_core_web_sm")


def spacy_tokenize(x):
    return [
        tok.text
        for tok in spacy.tokenizer(x)
        if not tok.is_punct | tok.is_space
    ]


def filter_long(s):
    return len(s.text) < 100


def test_dataset_preprocess():
    field = Field(
        tokenizer=spacy_tokenize,
        is_sequential=True,
        to_lower=True,
        eos_token="<eos>",
        sos_token="<sos>",
    )
    fields = {"text": field}
    with open(
        f"{os.path.dirname(os.path.realpath(__file__))}/test_data.txt",
        "r",
        encoding="utf-8",
    ) as f:
        data = [l.strip() for l in f][:50]
    examples = [Example.fromlist([example], fields) for example in data]
    ds = Dataset(examples, fields)
    assert len(ds) == len(data)
    assert isinstance(ds[0], Example)
    assert isinstance(ds[0].text, list)
    # test iteration
    for ex in ds:
        assert ex == ds[0]
        break
    # test get attr
    for ex in ds.text:
        assert ex == ds[0].text
        break


def test_filter_pred():
    field = Field(
        tokenizer=spacy_tokenize,
        is_sequential=True,
        to_lower=True,
        eos_token="<eos>",
        sos_token="<sos>",
    )
    fields = {"text": field}
    with open(
        f"{os.path.dirname(os.path.realpath(__file__))}/test_data.txt",
        "r",
        encoding="utf-8",
    ) as f:
        data = [l.strip() for l in f]
    examples = [Example.fromlist([example], fields) for example in data]
    ds = Dataset(examples, fields, filter_pred=filter_long)
    for ex in ds.text:
        assert len(ex) < 100


def test_split():
    text = Field(
        tokenizer=spacy_tokenize,
        is_sequential=True,
        to_lower=False,
        eos_token="<eos>",
        sos_token="<sos>",
    )
    label = Field(
        tokenizer=spacy_tokenize,
        is_sequential=True,
        is_target=True,
        to_lower=False,
    )
    fields = {"text": text, "label": label}

    data = [["a happy comment", "positive"]] * 100 + [
        ["sad comment", "negative"]
    ] * 10

    examples = [Example.fromlist(example, fields) for example in data]
    ds = Dataset(examples, fields)
    train_examples, test_examples, val_examples = ds.split(
        [0.8, 0.1, 0.1], stratify_field="label"
    )

    # check portion
    assert len(train_examples) == len(data) * 0.8
    assert len(test_examples) == len(data) * 0.1
    assert len(val_examples) == len(data) * 0.1
