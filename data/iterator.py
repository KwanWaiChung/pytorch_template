from ..data.batcher import Batcher
from multiprocessing.pool import Pool
import math


class Iterator:
    """Defines an iterator that loads batches of data from a Dataset

    It is iterable. We can loop over the batches. Each batch returns
    multiple `torch.Tensor` which corresponds to `self.dataset.fields`.
    If `include_lengths` is True, the `length` Tensor will be given
    after that field (X, len_X, y, len_y).

    Attributes:
        dataset (Dataset): The Dataset object to load Examples from.
        batch_size (int): Batch size.
        seed (int): The random seed used for shuffling.
        batcher (Batcher): An util object that helps to batch the examples.
            It shuffles and sorts the batches if necessary.

    Examples:
        >>> ds = Dataset(examples=examples, fields=dict(fields))
        >>> it = Iterator(dataset=ds, batch_size=4)
        >>> for train_X, train_y in it:
        ...     print(train_X.shape)
        (batch_size, seq_len, feature_dim)

    """

    def __init__(
        self,
        dataset: "Dataset",
        batch_size: int,
        seed: int = 0,
        sort_within_batch: bool = False,
        to_shuffle: bool = False,
    ):
        """Defines an iterator that loads batches of data from a Dataset

        Args:
            dataset (Dataset): The Dataset object to load Examples from.
            batch_size (int): Batch size.
            seed (int): The random seed used for shuffling.
            sort_within_batch (bool): If True, sort the data in
                each mini-batch.
            to_shuffle (bool): If True, shuffle the data for every epoch.
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
