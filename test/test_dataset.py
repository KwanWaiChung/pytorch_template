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
    fields = [("text", field)]
    with open(
        f"{os.path.dirname(os.path.realpath(__file__))}/test_data.txt",
        "r",
        encoding="utf-8",
    ) as f:
        data = [l.strip() for l in f][:50]
    examples = [Example.fromlist([example], fields) for example in data]
    ds = Dataset(examples, dict(fields))
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
    fields = [("text", field)]
    with open(
        f"{os.path.dirname(os.path.realpath(__file__))}/test_data.txt",
        "r",
        encoding="utf-8",
    ) as f:
        data = [l.strip() for l in f]
    examples = [Example.fromlist([example], fields) for example in data]
    ds = Dataset(examples, dict(fields), filter_pred=filter_long)
    for ex in ds.text:
        assert len(ex) < 100


#  def test_speed():
#  transform = [("wait", func)]
#  field = Field(preprocessing=transform, is_sequential=True, tokenizer=None)
#  fields = [("text", field)]
#  examples = ["hello"] * 10
#  examples = [Example.fromlist([example], fields) for example in examples]
#  fields = dict(fields)
#  ds = Dataset(examples, fields)
#  return True
