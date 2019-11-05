from ..utils.encoder import LabelEncoder, TextEncoder
from .mock_handler import MockLoggingHandler
from ..utils.logger import getlogger


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
