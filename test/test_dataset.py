from ..data import Dataset, TabularDataset, Example, Field
from ..utils import spacy_tokenize
from .utils.dataset import getTextData, getCsvFilename, getJsonFilename
import os


def filter_long(s):
    return len(s.text) < 100


DATA = getTextData()


def test_dataset_preprocess():
    field = Field(
        tokenizer=spacy_tokenize,
        is_sequential=True,
        to_lower=True,
        eos_token="<eos>",
        sos_token="<sos>",
    )
    fields = {"text": field}
    examples = [Example.fromlist([example], fields) for example in DATA]
    ds = Dataset(examples, fields)
    assert len(ds) == len(DATA)
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
    examples = [Example.fromlist([example], fields) for example in DATA]
    ds = Dataset(examples, fields, filter_pred=filter_long)
    for ex in ds.text:
        assert len(ex) < 100


def test_split_with_list_of_ratios():
    text = Field(
        tokenizer=spacy_tokenize,
        is_sequential=False,
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


def test_split_with_train_ratio():
    text = Field(
        tokenizer=spacy_tokenize,
        is_sequential=True,
        to_lower=False,
        eos_token="<eos>",
        sos_token="<sos>",
    )
    label = Field(
        tokenizer=spacy_tokenize,
        is_sequential=False,
        is_target=True,
        to_lower=False,
    )
    fields = {"text": text, "label": label}

    data = [["a happy comment", "positive"]] * 100 + [
        ["sad comment", "negative"]
    ] * 10

    examples = [Example.fromlist(example, fields) for example in data]
    ds = Dataset(examples, fields)
    train_ds, test_ds = ds.split(0.8, stratify_field="label")

    # check portion
    assert len(train_ds.examples) == len(data) * 0.8
    assert len(test_ds.examples) == len(data) * 0.2


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
        path=getCsvFilename(),
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
        path=getJsonFilename(),
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
        path=getCsvFilename(),
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
        path=getJsonFilename(),
        format="json",
        fields={"text": text, "stars": label},
        reader_params={"encoding": "utf-8"},
    )
    for example in ds:
        assert type(example.text) == list
        assert type(example.stars) == int


def test_tabular_dataset_with_json_filter_pred():
    text = Field(
        tokenizer=spacy_tokenize,
        is_sequential=True,
        to_lower=True,
        fix_length=150,
        batch_first=True,
    )
    label = Field(is_sequential=False, is_target=True, to_lower=False)
    ds = TabularDataset(
        path=getJsonFilename(),
        format="json",
        fields={"text": text, "stars": label},
        reader_params={"encoding": "utf-8"},
        filter_pred=lambda x: x.stars == 1 or x.stars == 5,
    )
    for example in ds:
        assert example.stars == 1 or example.stars == 5


def test_tabular_dataset_with_json_postprocessor():
    text = Field(
        tokenizer=spacy_tokenize,
        is_sequential=True,
        to_lower=True,
        fix_length=150,
        batch_first=True,
    )
    label = Field(is_sequential=False, is_target=True, to_lower=False)

    def postprocessor(example):
        example.stars = 1 if example.stars > 3 else 0
        return example

    ds = TabularDataset(
        path=getJsonFilename(),
        format="json",
        fields={"text": text, "stars": label},
        reader_params={"encoding": "utf-8"},
        postprocessor=postprocessor,
    )

    for example in ds:
        assert example.stars == 0 or example.stars == 1


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
        path=getCsvFilename(),
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
