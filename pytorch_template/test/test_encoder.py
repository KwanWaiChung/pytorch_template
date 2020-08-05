from ..utils.encoder import LabelEncoder, TextEncoder
from .utils.stub.mock_handler import MockLoggingHandler
from ..utils.logger import getlogger
from ..exceptions.NotFittedError import NotFittedError
import pytest
import os


handler = MockLoggingHandler(level="DEBUG")
logger = getlogger()
logger.parent.addHandler(handler)


def test_label_encoder_with_1d_vector():
    encoder = LabelEncoder()
    y = ["apple", "orange", "pears", "apple", "watermellon", "orange", "apple"]
    encoder.fit(y)
    assert len(set(y)) == len(encoder.stoi)
    assert encoder.decode(encoder.encode(y)) == y


def test_label_encoder_with_2d_vector():
    encoder = LabelEncoder()
    y = [
        ["apple"],
        ["orange"],
        ["pears"],
        ["apple"],
        ["watermellon"],
        ["orange"],
        ["apple"],
    ]
    encoder.fit(y)
    handler.messages["info"][-1] == (
        "A column-vector y was passed when a 1d array was"
        " expected. Please change the shape of y to "
        "(n_samples, ), for example using ravel()."
        " The column-vector is converted to a 1d array"
    )
    assert len(set([temp[0] for temp in y])) == len(encoder.stoi)
    assert encoder.decode(encoder.encode(y)) == [temp[0] for temp in y]


def test_label_encoder_encode_not_fitted():
    encoder = LabelEncoder()
    with pytest.raises(NotFittedError) as e:
        encoder.encode(["apple"])
    assert "Encoder is not fitted" in str(e.value)


def test_label_encoder_decode_not_fitted():
    encoder = LabelEncoder()
    with pytest.raises(NotFittedError) as e:
        encoder.decode([1])
    assert "Encoder is not fitted" in str(e.value)


def test_label_encoder_pickle():
    encoder = LabelEncoder()
    y = ["apple", "orange", "pears", "apple", "watermellon", "orange", "apple"]
    encoder.fit(y)
    stoi = encoder.stoi
    stof = encoder.stof
    itos = encoder.itos
    encoder.dump("test_encoder.pickle")

    new_encoder = LabelEncoder.load("test_encoder.pickle")
    assert new_encoder.stoi == stoi
    assert new_encoder.stof == stof
    assert new_encoder.itos == itos
    os.remove("test_encoder.pickle")


def test_text_encoder():
    encoder = TextEncoder(
        sos_token="<sos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    y = [
        [
            "apple",
            "orange",
            "pears",
            "apple",
            "watermellon",
            "orange",
            "apple",
        ],
        [
            "apple",
            "orange",
            "pears",
            "apple",
            "watermellon",
            "orange",
            "apple",
        ],
    ]
    encoder.fit(y)
    assert len(set(y[0])) + 4 == len(encoder.stoi)
    for sent in y:
        assert encoder.decode(encoder.encode(sent)) == sent

    assert encoder.unk_id == encoder.stoi["<unk>"]
    assert "<unk>" == encoder.itos[encoder.unk_id]
    assert encoder.sos_id == encoder.stoi["<sos>"]
    assert "<sos>" == encoder.itos[encoder.sos_id]
    assert encoder.eos_id == encoder.stoi["<eos>"]
    assert "<eos>" == encoder.itos[encoder.eos_id]
    assert encoder.pad_id == encoder.stoi["<pad>"]
    assert "<pad>" == encoder.itos[encoder.pad_id]

    assert encoder.stof["apple"] == 6, "The word frequency is incorrect"

    encoder = TextEncoder(
        sos_token="<sos>", pad_token="<pad>", unk_token="<unk>"
    )
    y = [
        [
            "apple",
            "orange",
            "pears",
            "apple",
            "watermellon",
            "orange",
            "apple",
        ],
        [
            "apple",
            "orange",
            "pears",
            "apple",
            "watermellon",
            "orange",
            "apple",
        ],
    ]
    encoder.fit(y)
    assert len(set(y[0])) + 3 == len(encoder.stoi)
    for sent in y:
        assert encoder.decode(encoder.encode(sent)) == sent

    assert encoder.unk_id == encoder.stoi["<unk>"]
    assert "<unk>" == encoder.itos[encoder.unk_id]
    assert encoder.sos_id == encoder.stoi["<sos>"]
    assert "<sos>" == encoder.itos[encoder.sos_id]
    assert encoder.pad_id == encoder.stoi["<pad>"]
    assert "<pad>" == encoder.itos[encoder.pad_id]

    assert encoder.stof["apple"] == 6, "The word frequency is incorrect"


def test_text_encoder_pickle():
    encoder = TextEncoder(
        sos_token="<sos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    y = [
        [
            "apple",
            "orange",
            "pears",
            "apple",
            "watermellon",
            "orange",
            "apple",
        ],
        [
            "apple",
            "orange",
            "pears",
            "apple",
            "watermellon",
            "orange",
            "apple",
        ],
    ]
    encoder.fit(y)
    stoi = encoder.stoi
    stof = encoder.stof
    itos = encoder.itos
    pad_id = encoder.pad_id
    encoder.dump("test_encoder.pickle")

    new_encoder = TextEncoder.load("test_encoder.pickle")
    assert new_encoder.stoi == stoi
    assert new_encoder.stof == stof
    assert new_encoder.itos == itos
    assert new_encoder.sos_token == "<sos>"
    assert new_encoder.eos_token == "<eos>"
    assert new_encoder.pad_token == "<pad>"
    assert new_encoder.unk_token == "<unk>"
    assert new_encoder.pad_id == pad_id
    os.remove("test_encoder.pickle")

    encoder = TextEncoder(
        sos_token="<sos>", pad_token="<pad>", unk_token="<unk>"
    )
    y = [
        [
            "apple",
            "orange",
            "pears",
            "apple",
            "watermellon",
            "orange",
            "apple",
        ],
        [
            "apple",
            "orange",
            "pears",
            "apple",
            "watermellon",
            "orange",
            "apple",
        ],
    ]
    encoder.fit(y)
    stoi = encoder.stoi
    stof = encoder.stof
    itos = encoder.itos
    encoder.dump("test_encoder.pickle")

    new_encoder = TextEncoder.load("test_encoder.pickle")
    assert new_encoder.stoi == stoi
    assert new_encoder.stof == stof
    assert new_encoder.itos == itos
    assert new_encoder.sos_token == "<sos>"
    assert new_encoder.eos_token is None
    assert new_encoder.pad_token == "<pad>"
    assert new_encoder.unk_token == "<unk>"
    os.remove("test_encoder.pickle")


def test_text_encoder_max_size():
    encoder = TextEncoder(
        sos_token=None,
        eos_token=None,
        pad_token=None,
        unk_token=None,
        max_size=3,
    )
    y = [
        [
            "apple",
            "orange",
            "pears",
            "apple",
            "watermellon",
            "orange",
            "apple",
        ],
        [
            "apple",
            "orange",
            "pears",
            "apple",
            "watermellon",
            "orange",
            "apple",
        ],
    ]
    encoder.fit(y)
    assert len(encoder.stoi) == 3
    encoder = TextEncoder(
        sos_token=None, eos_token=None, pad_token=None, unk_token=None
    )
    encoder.fit(y)
    assert len(encoder.stoi) == 4


def test_text_encoder_min_freq():
    encoder = TextEncoder(
        sos_token=None,
        eos_token=None,
        pad_token=None,
        unk_token=None,
        min_freq=4,
    )
    y = [
        [
            "apple",
            "orange",
            "pears",
            "apple",
            "watermellon",
            "orange",
            "apple",
        ],
        [
            "apple",
            "orange",
            "pears",
            "apple",
            "watermellon",
            "orange",
            "apple",
        ],
    ]
    encoder.fit(y)
    assert len(encoder.stoi) == 2
