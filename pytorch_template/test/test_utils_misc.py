from ..utils.misc import batch


def test_batch():
    examples = [1, 2, 3, 4, 5, 6, 7]
    for i, minibatch in enumerate(batch(examples, 3)):
        if i == 0:
            assert minibatch == [1, 2, 3]
        elif i == 1:
            assert minibatch == [4, 5, 6]
        if i == 2:
            assert minibatch == [7]
