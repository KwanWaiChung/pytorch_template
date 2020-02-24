from time import sleep
from random import random
from ..data.field import Field
from ..data.dataset import Dataset, TabularDataset
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
    train_ds, test_ds, val_ds = ds.split(
        [0.8, 0.1, 0.1], stratify_field="label"
    )

    # check portion
    assert len(train_ds.examples) == len(data) * 0.8
    assert len(test_ds.examples) == len(data) * 0.1
    assert len(val_ds.examples) == len(data) * 0.1


def test_read_csv():
    text = Field(
        tokenizer=spacy_tokenize,
        is_sequential=True,
        to_lower=True,
        fix_length=150,
        batch_first=True,
    )
    label = Field(
        tokenizer=spacy_tokenize,
        is_sequential=False,
        is_target=True,
        to_lower=False,
    )
    examples = TabularDataset.read_csv(
        path=f"{os.path.dirname(os.path.realpath(__file__))}/train.csv",
        fields={"comment_text": text, "toxic": label, "severe_toxic": label},
        params={"encoding": "utf-8"},
    )
    assert type(examples[0].comment_text) == str
    assert type(examples[0].toxic) == int
    assert type(examples[0].severe_toxic) == int


def test_read_json():
    text = Field(
        tokenizer=spacy_tokenize,
        is_sequential=True,
        to_lower=True,
        fix_length=150,
        batch_first=True,
    )
    label = Field(
        tokenizer=spacy_tokenize,
        is_sequential=False,
        is_target=True,
        to_lower=False,
    )
    examples = TabularDataset.read_json(
        path=f"{os.path.dirname(os.path.realpath(__file__))}/reviews.json",
        fields={"text": text, "stars": label},
        params={"encoding": "utf-8"},
    )
    assert type(examples[0].text) == str
    assert type(examples[0].stars) == int


def test_tabular_dataset_with_csv():
    text = Field(
        tokenizer=spacy_tokenize,
        is_sequential=True,
        to_lower=True,
        fix_length=150,
        batch_first=True,
    )
    label = Field(is_sequential=False, is_target=True, to_lower=False)
    ds = TabularDataset(
        path=f"{os.path.dirname(os.path.realpath(__file__))}/train.csv",
        format="csv",
        fields={"comment_text": text, "toxic": label, "severe_toxic": label},
        reader_params={"encoding": "utf-8"},
    )
    for example in ds:
        assert type(example.comment_text) == list
        assert type(example.toxic) == int
        assert type(example.severe_toxic) == int


def test_tabular_dataset_with_json():
    text = Field(
        tokenizer=spacy_tokenize,
        is_sequential=True,
        to_lower=True,
        fix_length=150,
        batch_first=True,
    )
    label = Field(is_sequential=False, is_target=True, to_lower=False)
    ds = TabularDataset(
        path=f"{os.path.dirname(os.path.realpath(__file__))}/reviews.json",
        format="json",
        fields={"text": text, "stars": label},
        reader_params={"encoding": "utf-8"},
    )
    for example in ds:
        assert type(example.text) == list
        assert type(example.stars) == int


def sort_key(example):
    return example.comment_text


def test_pickle():
    text = Field(
        tokenizer=spacy_tokenize,
        is_sequential=True,
        to_lower=True,
        fix_length=150,
        batch_first=True,
    )
    label = Field(is_sequential=False, is_target=True, to_lower=False)
    ds = TabularDataset(
        path=f"{os.path.dirname(os.path.realpath(__file__))}/train.csv",
        format="csv",
        fields={"comment_text": text, "toxic": label, "severe_toxic": label},
        reader_params={"encoding": "utf-8"},
        sort_key=sort_key,
    )

    examples = ds.examples
    fields = ds.fields
    ds.dump("ds.pickle")
    ds = Dataset.load("ds.pickle")
    assert ds.examples == examples
    assert ds.fields == fields
    assert ds.sort_key == sort_key
    os.remove("ds.pickle")
