from ..data.iterator import Iterator
from ..data.field import Field
from ..data.example import Example
from ..data.dataset import Dataset
from ..utils.encoder import TextEncoder
import pytest
import torch


def test_nonexist_field_name():
    field = Field(
        is_sequential=True, to_lower=True, eos_token="<eos>", sos_token="<sos>"
    )
    fields = [("text", field)]
    examples = [Example.fromlist(["Hello world"], fields)]
    ds = Dataset(examples, dict(fields))
    with pytest.raises(ValueError) as e:
        iterator = Iterator(["non_exist"], ds, 4, 0, True, True)
    assert str(e.value) == "The dataset has no field named `non_exist`"


def test_iteration():
    data = [
        [
            "Some english sentence",
            "A french tranlsation maybe but you got the idea",
        ],
        ["Hello world, its enlish", "bon appeti"],
        [
            "Some english sentence",
            "A french tranlsation maybe but you got the idea",
        ],
        ["Hello world, its enlish", "bon appeti"],
        [
            "Some english sentence",
            "A french tranlsation maybe but you got the idea",
        ],
        ["Hello world, its enlish", "bon appeti"],
        [
            "Some english sentence",
            "A french tranlsation maybe but you got the idea",
        ],
        ["Hello world, its enlish", "bon appeti"],
        [
            "Some english sentence",
            "A french tranlsation maybe but you got the idea",
        ],
        ["Hello world, its enlish", "bon appeti"],
    ]
    eng_field = Field(
        is_sequential=True, to_lower=True, eos_token="<eos>", sos_token="<sos>"
    )
    fre_field = Field(
        is_sequential=True, to_lower=True, eos_token="<eos>", sos_token="<sos>"
    )
    fields = [("eng", eng_field), ["fre", fre_field]]
    examples = [Example.fromlist(example, fields) for example in data]
    ds = Dataset(examples, dict(fields), sort_key=lambda x: len(x.eng))
    eng_encoder = TextEncoder(
        sos_token="<sos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    fre_encoder = TextEncoder(
        sos_token="<sos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    eng_field.build_vocab(eng_encoder, ds)
    fre_field.build_vocab(fre_encoder, ds)

    iterator = Iterator(
        ["fre", "eng"],
        dataset=ds,
        batch_size=4,
        seed=0,
        sort_within_batch=True,
        to_shuffle=False,
    )

    import pdb

    for (f, e) in iterator:
        assert type(f) == torch.Tensor
        assert type(e) == torch.Tensor
        pdb.set_trace()
