from ..utils.misc import pad

test_str = ["This", "is", "a", "sentence", "."]


def test_padding_with_no_tokens():
    assert pad(test_str, max_len=10) == (
        test_str + ["<pad>"] * (10 - len(test_str))
    )


def test_padding_with_sos():
    padded = pad(test_str, max_len=10, sos_token="<sos>")
    assert len(padded) == 11
    assert padded == ["<sos>"] + test_str + ["<pad>"] * (10 - len(test_str))


def test_padding_with_eos():
    padded = pad(test_str, max_len=10, eos_token="<eos>")
    assert len(padded) == 11
    assert padded == test_str + ["<eos>"] + ["<pad>"] * (10 - len(test_str))


def test_padding_with_sos_eos():
    padded = pad(test_str, max_len=10, sos_token="<sos>", eos_token="<eos>")
    assert len(padded) == 12
    assert padded == ["<sos>"] + test_str + ["<eos>"] + ["<pad>"] * (
        10 - len(test_str)
    )


def test_padding_first():
    assert pad(test_str, max_len=10, pad_first=True) == (
        ["<pad>"] * (10 - len(test_str)) + test_str
    )


def test_padding_last():
    assert pad(test_str, max_len=10, pad_first=False) == (
        test_str + ["<pad>"] * (10 - len(test_str))
    )


def test_truncate_first():
    assert pad(test_str, max_len=2, truncate_first=True) == test_str[-2:]
    assert pad(
        test_str,
        max_len=2,
        truncate_first=True,
        sos_token="<sos>",
        eos_token="<eos>",
    ) == ["<sos>"] + test_str[-2:] + ["<eos>"]


def test_truncate_last():
    assert pad(test_str, max_len=2, truncate_first=False) == test_str[:2]
    assert pad(
        test_str,
        max_len=2,
        truncate_first=False,
        sos_token="<sos>",
        eos_token="<eos>",
    ) == ["<sos>"] + test_str[:2] + ["<eos>"]
