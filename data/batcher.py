from .example import Example
from typing import Callable, List
from ..utils.misc import batch
import math
import random


class Batcher:
    """This class return batches of Examples.

    The examples will be shuffled and sorted if needed. The returned
    Examples are not processed (pad and numericalize) yet.

    Attributes:

    """

    def __init__(
        self,
        examples: List[Example],
        batch_size: int,
        seed: int = 0,
        sort_key: Callable[["Example"], int] = None,
        sort_within_batch: bool = None,
        to_shuffle: bool = None,
    ):
        """
        Args:
            examples: The full List of examples.
            batch_size: Batch size.
            seed: The seed for random shuffling.
            sort_key: A key to use for sorting examples. Usually
                its the length of some attributes.
            sort_within_batch: Whether to sort (in descending order)
                within each batch.
            to_shuffle: Whter to shuffle examples between epochs.
        """
        self.examples = examples
        self.batch_size = batch_size
        self._random = random.Random(seed)
        self.sort_key = sort_key
        self.sort_within_batch = sort_within_batch
        self.to_shuffle = to_shuffle

    def _shuffle_examples(self):
        if self.to_shuffle:
            return self._random.sample(self.examples, len(self.examples))
        else:
            return self.examples

    def __iter__(self):
        examples = self._shuffle_examples()
        for minibatch in batch(examples, self.batch_size):
            if self.sort_within_batch:
                minibatch = sorted(minibatch, key=self.sort_key, reverse=True)
            yield minibatch

    def __len__(self):
        return math.ceil(len(self.examples) / self.batch_size)


class BucketBatcher(Batcher):
    def __iter__(self):
        if not self.sort_within_batch:
            return super().__iter__()
        examples = self._shuffle_examples()
        for bucket in batch(examples, self.batch_size * 100):
            bucket = sorted(bucket, key=self.sort_key, reverse=True)
            minibatches = list(batch(bucket, self.batch_size))
            if self.to_shuffle:
                minibatches = self._random.sample(
                    minibatches, len(minibatches)
                )
            for minibatch in minibatches:
                yield minibatch
