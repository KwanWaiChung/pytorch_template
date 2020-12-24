from allennlp.data import DatasetReader, Instance, AllennlpDataset
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import TextField, LabelField, MetadataField
from typing import Dict, List, Callable, Union, Iterable, Tuple
from ..data import Pipeline
from ..utils.googledrive_downloader import download_from_googledrive
from ..utils.misc import get_split_ratio, split as _split
import os

ID = {"imdb": "1RmAPdqCG69UFnhL8--PPzi7TlcNcBV1C"}


class TextClassificationDataset(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        text_transform: List[Callable[[str], str]] = None,
        label_transform: List[Callable[[str], str]] = None,
        max_len: int = None,
        lazy: bool = False,
        cache_directory: str = None,
    ):
        super().__init__(lazy=lazy, cache_directory=cache_directory)
        self.tokenizer = tokenizer or SpacyTokenizer()
        self.token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(
                namespace="tokens",
            )
        }
        self.text_transform = Pipeline(text_transform)
        self.label_transform = Pipeline(label_transform)
        self.max_len = max_len

    def _read(self, file_path: str) -> Iterable[Instance]:
        raise NotImplementedError

    def text_to_instance(self, text: str, label: Union[str, int]) -> Instance:
        fields = {}
        fields["raw_text"] = MetadataField(text)
        text = self.text_transform(text)
        label = self.label_transform(label)
        tokens = self.tokenizer.tokenize(text)
        if self.max_len:
            tokens = tokens[: self.max_len]
        fields["text"] = TextField(tokens, self.token_indexers)
        fields["label"] = LabelField(
            label, label_namespace="labels", skip_indexing=type(label) == int
        )
        return Instance(fields)

    def split(
        self,
        dataset: AllennlpDataset,
        split_ratio: Union[List[float], float, int],
        seed: int = 0,
    ) -> Tuple[AllennlpDataset, AllennlpDataset]:
        stratify = [instance["label"].label for instance in dataset]
        return _split(dataset, split_ratio, stratify=stratify, seed=seed)


class ImdbDataset(TextClassificationDataset):
    """
    Imdb sentiment dataset. It contains movie reviews and each of them
    are either posotive or negative.

    """

    def __init__(
        self, file_path: str, n_examples: int = float("inf"), **kwargs
    ):
        """
        Args:
            file_path: The root folder for the dataset. It will download
                from drive if folder doesn't exist.

        """
        self.n_examples = n_examples
        self.file_path = file_path
        os.makedirs(file_path, exist_ok=True)
        if "train-pos.txt" not in os.listdir(file_path):
            download_from_googledrive(ID["imdb"], dst_dir=file_path)
        super().__init__(**kwargs)

    def _read(self, mode: str) -> Iterable[Instance]:
        """
        file_path: The root directory path.
            file_path/
                train-pos.txt
                train-neg.txt
                test-pos.txt
                test-neg.txt

        Args:
            mode: Either `train` or `test`.

        """
        for filename in os.listdir(self.file_path):
            if mode not in filename:
                continue
            with open(os.path.join(self.file_path, filename)) as f:
                for i, line in enumerate(f):
                    if i >= self.n_examples:
                        return
                    yield self.text_to_instance(
                        line.strip(), "pos" if "pos" in filename else "neg"
                    )
