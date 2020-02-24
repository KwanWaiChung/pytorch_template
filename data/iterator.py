from ..data.batcher import Batcher
from multiprocessing.pool import Pool
import math


class Iterator:
    """Defines an iterator that loads batches of data from a Dataset

    It is iterable. We can loop over the batches. Each batch returns
    few `torch.Tensor` which corresponds to `self.dataset.fields`.
    If `include_lengths` is True, the `length` Tensor will be given
    after that field (X, len_X, y, len_y).
        >>> ds = Dataset(examples=examples, fields=dict(fields))
        >>> it = Iterator(dataset=ds, batch_size=4)
        >>> for train_X, train_y in it:
        ...     print(train_X.shape)
        (batch_size, seq_len, feature_dim)

    Attributes:
        dataset (Dataset): The Dataset object to load Examples from.
        batch_size (int): Batch size.
        seed (int): The random seed used for shuffling.
        batcher (Batcher): An util object that helps to batch the examples.
            It shuffles and sorts the batches if necessary.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        seed=0,
        sort_within_batch=False,
        to_shuffle=False,
    ):
        """Defines an iterator that loads batches of data from a Dataset

        Attributes:
            dataset (Dataset): The Dataset object to load Examples from.
            batch_size (int): Batch size.
            seed (int): The random seed used for shuffling.
            batch(Batcher): An util object that helps to shuffle and
                sort if necessary.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.batcher = Batcher(
            dataset, batch_size, seed, sort_within_batch, to_shuffle
        )

    def __iter__(self):
        for minibatch in self.batcher:
            result = []
            for field_name, field in self.dataset.fields.items():
                batch = [getattr(example, field_name) for example in minibatch]
                batch = field.process(batch)
                if field.include_lengths:
                    result += batch
                else:
                    result.append(batch)
            yield result

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
