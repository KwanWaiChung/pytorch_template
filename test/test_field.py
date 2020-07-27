from ..data.example import Example
from ..data.field import Field
from ..data.dataset import Dataset
from ..utils.encoder import TextEncoder, LabelEncoder
from .utils.dataset import getTextData
import re

DATA = getTextData()


def func1(s: str) -> str:
    return s.lower()


def func2(s: str) -> str:
    return s[:3]


def test_preprocess():
    transform = [("lower", func1), ("trim3", func2)]

    field = Field(preprocessing=transform, is_sequential=True, tokenizer=None)
    assert field.preprocess(["HELLO"]) == ["hel"]


def test_cleaning():
    transform = [("lower", func1), ("trim3", func2)]

    def cleaner(s):
        return re.sub("<.*?>", "", s)

    field = Field(
        cleaning=[("remove_punc", cleaner)],
        preprocessing=transform,
        is_sequential=True,
        tokenizer=None,
    )
    assert field.preprocess(["<a>HELLO</a>"]) == ["hel"]


def test_lower():
    transform = [("trim3", func2)]
    field1 = Field(
        preprocessing=transform,
        to_lower=True,
        is_sequential=True,
        tokenizer=None,
    )
    field2 = Field(preprocessing=transform, to_lower=True, is_sequential=True)
    assert field1.preprocess(["HELLO"]) == [
        "hel"
    ], "parameter `lower` is not working"
    assert field2.preprocess("HELLO WOrld") == ["hel", "wor"]


def test_sequential():
    transform = [("lower", func1), ("trim3", func2)]
    field1 = Field(preprocessing=transform, is_sequential=True)
    field2 = Field(preprocessing=transform, is_sequential=False)
    assert field1.preprocess("HELLO WOrld") == ["hel", "wor"]
    assert field2.preprocess(["HELLO", "WOrld"]) == ["hel", "wor"]


def test_stopwords():
    field1 = Field(to_lower=True, is_sequential=True, stopwords=["the"])
    field2 = Field(
        to_lower=True, is_sequential=True, stopwords=["the"], tokenizer=None
    )

    assert field1.preprocess("the Apple pie") == ["apple", "pie"]
    assert field2.preprocess(["the", "Apple", "pie"]) == ["apple", "pie"]


def test_pad():
    field = Field(
        eos_token="<eos>",
        sos_token="<sos>",
        fix_length=5,
        is_sequential=True,
        tokenizer=None,
    )
    sentence = [
        ["hello", "world", "!", "this", "is", "me", "."],
        ["short", "sentence"],
    ]
    result = field._pad(sentence)
    assert result[0] == ["<sos>"] + sentence[0][:5] + ["<eos>"]
    assert result[1] == ["<sos>"] + sentence[1] + ["<eos>"] + ["<pad>"] * (
        5 - len(sentence[1])
    )


def test_pad_length():
    field = Field(
        eos_token="<eos>",
        sos_token="<sos>",
        fix_length=5,
        is_sequential=True,
        tokenizer=None,
        include_lengths=True,
    )
    sentence = [
        ["hello", "world", "!", "this", "is", "me", "."],
        ["short", "sentence"],
    ]
    result = field._pad(sentence)
    assert result[1][0] == 7
    assert result[1][1] == 4


def test_build_vocab_with_one_dataset():
    field = Field(
        eos_token="<eos>",
        sos_token="<sos>",
        fix_length=5,
        is_sequential=True,
        include_lengths=True,
    )

    fields = {"text": field}
    examples = [Example.fromlist([example], fields) for example in DATA]
    ds = Dataset(examples, fields)
    encoder = field.build_vocab(ds)

    from collections import Counter

    c = Counter(sum(ds.text, []))
    for word, freq in c.items():
        assert encoder.stof[word] == freq, "frequency count isn't right"


def test_build_vocab_with_two_dataset():
    data1 = [
        "The company's hardware products include the iPhone smartphone",
        "I hate iPhone",
    ]
    data2 = ["I hate apple"]

    field = Field(
        eos_token="<eos>",
        sos_token="<sos>",
        fix_length=5,
        is_sequential=True,
        include_lengths=True,
    )

    fields = {"text": field}

    examples1 = [Example.fromlist([example], fields) for example in data1]
    examples2 = [Example.fromlist([example], fields) for example in data2]
    ds1 = Dataset(examples1, fields)
    ds2 = Dataset(examples2, fields)
    encoder = field.build_vocab(ds1, ds2)

    assert encoder.stof == {
        "The": 1,
        "company's": 1,
        "hardware": 1,
        "products": 1,
        "include": 1,
        "the": 1,
        "iPhone": 2,
        "smartphone": 1,
        "I": 2,
        "hate": 2,
        "apple": 1,
    }


def test_numericalize():
    data = [["I love iPhone", "pos"], ["I hate iPhone", "neg"]]
    text = Field(is_sequential=True)
    label = Field(is_sequential=False, is_target=True)

    fields = {"text": text, "label": label}
    examples = [Example.fromlist(example, fields) for example in data]

    ds = Dataset(examples, fields)
    text_encoder = text.build_vocab(ds)
    label_encoder = label.build_vocab(ds)

    text_digit = text._numericalize_batch(ds.text)
    # decode return [label]
    assert ds[0].text == text_encoder.decode(text_digit[0])
    assert ds[1].text == text_encoder.decode(text_digit[1])

    label_digit = label._numericalize_batch(ds.label)
    # decode return [label]
    assert data[0][1] == label_encoder.decode(label_digit[0])[0]
    assert data[1][1] == label_encoder.decode(label_digit[1])[0]


def test_process():
    field = Field(
        eos_token="<eos>", sos_token="<sos>", fix_length=5, is_sequential=True
    )

    fields = {"text": field}
    examples = [Example.fromlist([example], fields) for example in DATA]
    ds = Dataset(examples, fields)
    encoder = field.build_vocab(
        ds,
        sos_token="<sos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )

    batch = field.process(list(ds.text)[:4])
    for i, example in enumerate(batch):
        assert encoder.decode(example) == ["<sos>"] + ds[i].text[:5] + [
            "<eos>"
        ]


def test_process_with_length():
    field = Field(
        eos_token="<eos>",
        sos_token="<sos>",
        fix_length=5,
        is_sequential=True,
        include_lengths=True,
    )

    fields = [("text", field)]
    fields = {"text": field}
    examples = [Example.fromlist([example], fields) for example in DATA]
    ds = Dataset(examples, fields)
    encoder = field.build_vocab(
        ds,
        sos_token="<sos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )

    batch = field.process(list(ds.text)[:4])
    for i, (example, length) in enumerate(zip(*batch)):
        assert encoder.decode(example) == ["<sos>"] + ds[i].text[:5] + [
            "<eos>"
        ]
        assert length.item() == 7
