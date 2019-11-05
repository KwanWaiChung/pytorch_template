from ..data.example import Example
from ..data.field import Field
from ..data.batcher import Batcher, BucketBatcher
import numpy as np
import os
import random


def test_shuffle_examples():
    field = Field(
        is_sequential=True, to_lower=True, eos_token="<eos>", sos_token="<sos>"
    )
    fields = [("text", field)]
    with open(
        f"{os.path.dirname(os.path.realpath(__file__))}/test_data.txt",
        "r",
        encoding="utf-8",
    ) as f:
        data = [l.strip() for l in f][:50]
    examples = [Example.fromlist([example], fields) for example in data]

    batcher = Batcher(
        examples,
        batch_size=4,
        seed=0,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        to_shuffle=True,
    )
    new_examples = batcher._shuffle_examples()
    assert new_examples != examples

    batcher = Batcher(
        examples,
        batch_size=4,
        seed=0,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        to_shuffle=False,
    )
    new_examples = batcher._shuffle_examples()
    assert new_examples == examples


def test_iteration_with_no_shuffle_and_no_sort():
    field = Field(
        is_sequential=True, to_lower=True, eos_token="<eos>", sos_token="<sos>"
    )
    fields = [("text", field)]
    with open(
        f"{os.path.dirname(os.path.realpath(__file__))}/test_data.txt",
        "r",
        encoding="utf-8",
    ) as f:
        data = [l.strip() for l in f][:50]
    examples = [Example.fromlist([example], fields) for example in data]

    batcher = Batcher(
        examples,
        batch_size=4,
        seed=0,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        to_shuffle=False,
    )
    for i, batch in enumerate(batcher):
        assert batch == examples[i * 4 : i * 4 + 4]


def test_iteration_with_shuffle_and_no_sort():
    field = Field(
        is_sequential=True, to_lower=True, eos_token="<eos>", sos_token="<sos>"
    )
    fields = [("text", field)]
    with open(
        f"{os.path.dirname(os.path.realpath(__file__))}/test_data.txt",
        "r",
        encoding="utf-8",
    ) as f:
        data = [l.strip() for l in f][:50]
    examples = [Example.fromlist([example], fields) for example in data]

    batcher = Batcher(
        examples,
        batch_size=4,
        seed=0,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        to_shuffle=True,
    )

    random.seed(0)
    examples = random.sample(examples, len(examples))

    is_sorted = True
    for i, minibatch in enumerate(batcher):
        assert minibatch == examples[i * 4 : i * 4 + 4]
        length = [len(example.text) for example in minibatch]
        if length != sorted(length, reverse=True):
            is_sorted = False
    assert not is_sorted


def test_iteration_with_shuffle_and_sort():
    field = Field(
        is_sequential=True, to_lower=True, eos_token="<eos>", sos_token="<sos>"
    )
    fields = [("text", field)]
    with open(
        f"{os.path.dirname(os.path.realpath(__file__))}/test_data.txt",
        "r",
        encoding="utf-8",
    ) as f:
        data = [l.strip() for l in f][:50]
    examples = [Example.fromlist([example], fields) for example in data]

    batcher = Batcher(
        examples,
        batch_size=4,
        seed=0,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        to_shuffle=True,
    )
    batcher2 = Batcher(
        examples,
        batch_size=4,
        seed=0,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        to_shuffle=True,
    )

    import random

    random.seed(0)
    examples = random.sample(examples, len(examples))
    for i, minibatch in enumerate(batcher):
        length = len(minibatch[0].text)
        subexamples = examples[i * 4 : i * 4 + 4]
        # check length
        for example in minibatch:
            assert len(example.text) <= length
            assert example in subexamples
            length = len(example.text)


def test_seed():
    field = Field(
        is_sequential=True, to_lower=True, eos_token="<eos>", sos_token="<sos>"
    )
    fields = [("text", field)]
    with open(
        f"{os.path.dirname(os.path.realpath(__file__))}/test_data.txt",
        "r",
        encoding="utf-8",
    ) as f:
        data = [l.strip() for l in f][:50]
    examples = [Example.fromlist([example], fields) for example in data]
    batcher = Batcher(
        examples,
        batch_size=4,
        seed=0,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        to_shuffle=True,
    )
    batcher2 = Batcher(
        examples,
        batch_size=4,
        seed=0,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        to_shuffle=True,
    )
    batcher3 = Batcher(
        examples,
        batch_size=4,
        seed=1,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        to_shuffle=True,
    )
    is_different1 = False
    is_different2 = False
    for (minibatch1, minibatch2, minibatch3) in zip(
        batcher, batcher2, batcher3
    ):
        if minibatch1 != minibatch2:
            is_different1 = True
        if minibatch2 != minibatch3:
            is_different2 = True
    assert not is_different1
    assert is_different2


def test_bucket_batcher():
    field = Field(
        is_sequential=True,
        to_lower=True,
        eos_token="<eos>",
        sos_token="<sos>",
        tokenizer=None,
    )
    fields = [("text", field)]
    data = [["Hello", "world"]] * 50 + [["hi"]] * 50
    random.shuffle(data)
    examples = [Example.fromlist([example], fields) for example in data]

    bbatcher = BucketBatcher(
        examples,
        batch_size=50,
        seed=0,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        to_shuffle=True,
    )

    batcher = Batcher(
        examples,
        batch_size=50,
        seed=0,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        to_shuffle=True,
    )

    # test the minimize padding logic
    for i, (minibatch, minibbatch) in enumerate(zip(batcher, bbatcher)):
        assert not (
            np.array([len(example.text) for example in minibatch]) == 2
        ).all()
        assert (
            np.array([len(example.text) for example in minibbatch])
            == len(minibbatch[0].text)
        ).all()

    # test the minibatch shuffling
    data = [["hi"] * i for i in range(100)]
    examples = [Example.fromlist([example], fields) for example in data]
    bbatcher1 = BucketBatcher(
        examples,
        batch_size=2,
        seed=0,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        to_shuffle=True,
    )
    bbatcher2 = BucketBatcher(
        examples,
        batch_size=2,
        seed=0,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        to_shuffle=True,
    )
    bbatcher3 = BucketBatcher(
        examples,
        batch_size=2,
        seed=1024,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        to_shuffle=True,
    )

    is_different1 = False
    is_different2 = False
    for i, (minibatch1, minibatch2, minibatch3) in enumerate(
        zip(bbatcher1, bbatcher2, bbatcher3)
    ):
        assert len(minibatch1[0].text) - len(minibatch1[1].text) == 1
        assert len(minibatch2[0].text) - len(minibatch2[1].text) == 1
        assert len(minibatch3[0].text) - len(minibatch3[1].text) == 1
        if minibatch1 != minibatch2:
            is_different1 = True
        if minibatch1 != minibatch3:
            is_different2 = True
    assert not is_different1
    assert is_different2