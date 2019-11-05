from ..data.batcher import Batcher
from multiprocessing.pool import Pool


class Iterator:
    def __init__(
        self,
        field_names,
        dataset,
        batch_size,
        seed,
        sort_within_batch,
        to_shuffle,
    ):
        self.field_names = field_names
        self.dataset = dataset
        self.batch_size = batch_size
        self.batcher = Batcher(
            dataset.examples,
            batch_size,
            seed,
            dataset.sort_key,
            sort_within_batch,
            to_shuffle,
        )
        self.batch_size = batch_size
        self.seed = seed
        for field_name in field_names:
            if field_name not in dataset.fields:
                raise ValueError(
                    f"The dataset has no field named `{field_name}`"
                )

    def __iter__(self):
        for minibatch in self.batcher:
            result = []
            for field_name in self.field_names:
                field = self.dataset.fields[field_name]
                batch = [getattr(example, field_name) for example in minibatch]
                batch = field.process(batch)
                if field.include_lengths:
                    result += batch
                else:
                    result.append(batch)
            yield result
