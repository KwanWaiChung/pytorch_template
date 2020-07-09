from typing import Callable, List, Union, Tuple
from .pipeline import Pipeline
from ..utils.misc import pad as _pad
from ..utils.encoder import LabelEncoder, TextEncoder

#  from ..data.dataset import Dataset
from multiprocessing.pool import Pool
import torch


def naive_tokenizer(s: str) -> List[str]:
    return s.split()


class Field:
    """A processor that defines how to convert raw data to tensor.

    Attributes:
        cleaning: A `Pipeline` object that will be applied to the raw inputs
            of each example. Default: None.
        preprocessing: A `Pipeline` object that will be applied to the inputs
            of each example after tokenizing. Default: None.
        postprocessing: A `Pipeline` object that will be applied to the inputs
            of each example after numericalizing but before changing to
            tensors. Default: None.
        is_target: True if the feature is the target label. Default: False.
        tokenizer: The function used to tokenize strings.
        to_lower: True if lower the text in this field. Default: False.
        is_sequential: True if the data is sequential or a string.
            No tokenizing if False. Default: True.
        to_numericalize: if True, will convert the words to indexes.
            Default: True.
        include_lengths: If True, the list containing the lengths of each
            examples will be returned as well. Default: False.
        fix_length: A fixed length for all the examples.
        batch_first: If True, the batch size will be the first dimension
            in the returned tensor. Default: None.
        stopwords: Tokens to discard during preprocessing.
            Default: None.
        pad_token: The string token used as padding. Default: "<pad>"
        unk_token: The string token used as padding. Default: "<unk>"
        pad_first: If True, do the padding at the beginning. Default: False.
        truncate_first: If True, do the truncation at the beginning.
            Default: False.

    """

    def __init__(
        self,
        cleaning: List[Tuple[str, Callable[[str], str]]] = None,
        preprocessing: List[
            Tuple[str, Callable[[List[str]], List[str]]]
        ] = None,
        postprocessing: List[
            Tuple[str, Callable[[List[int]], List[int]]]
        ] = None,
        is_target: bool = False,
        tokenizer: Callable[[str], List[str]] = naive_tokenizer,
        to_lower: bool = False,
        is_sequential: bool = True,
        to_numericalize: bool = True,
        include_lengths: bool = False,
        fix_length: int = None,
        batch_first: bool = False,
        stopwords: List = [],
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        sos_token: str = None,
        eos_token: str = None,
        pad_first: bool = False,
        truncate_first: bool = False,
    ):
        self.cleaning = Pipeline(cleaning)
        self.preprocessing = Pipeline(preprocessing)
        self.postprocessing = Pipeline(postprocessing)
        self.is_target = is_target
        self.tokenizer = tokenizer
        self.to_lower = to_lower
        self.is_sequential = is_sequential
        self.to_numericalize = to_numericalize
        self.include_lengths = include_lengths
        self.fix_length = fix_length
        self.batch_first = batch_first
        self.stopwords = set(stopwords)
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_first = pad_first
        self.truncate_first = truncate_first

    def preprocess(self, s: Union[str, List[str]]) -> Union[str, List[str]]:
        """Preprocess a single example using this field, tokenizing if needed.

        Clean -> tokenize -> lower -> stopwords -> user defined preprocess
        The input will first pass to `cleaning`, like removing html tags.
        Then if `is_sequential` is True, it will be tokenized.
        Next, it will be converted to lower case (if needed).
        Stopwords will be also be removed . Finally, it will be passed to
        the user-provided preprocess pipeline.

        Args:
            s: It can be a sentence (str) or sequence of
                tokens (List[str])

        Returns:
            str or List[str]: List of str will be returned if either
                `is_sequential` is True or the input was already a sequence
                of tokens.

        Raises:
            TypeError: if input is a list but is_sequential is True and
                using a tokenizer.

        """
        if self.cleaning:
            s = self.cleaning(s)
        if self.is_sequential and self.tokenizer:
            if isinstance(s, list):
                raise TypeError(
                    "Only strings can be tokenized. "
                    "Either pass `None` to tokenizer or make sure "
                    "the input is a string."
                )
            s = self.tokenizer(s)  # List[str]
        if self.to_lower:
            s = Pipeline([("lower", str.lower)])(s)
        if self.stopwords:
            s = [token for token in s if token not in self.stopwords]
        if self.preprocessing:
            s = self.preprocessing(s)
        return s

    def process(self, batch: List[List[str]], to_tensor=True) -> torch.Tensor:
        """Do the padding and the numericalizing to a list of examples.

        Args:
            batch: A list of examples, where each example is a list of tokens.
            to_tensor: If True, a tensor will be returned. Otherwise, it will
                return List. Default: True.

        Returns:
             Tuple[List or torch.Tensor): If `include_lengths` is True, the
                length will be returned together in a tuple. Otherwise, it
                will just return the List or torch.Tensor.

        """
        batch = self._pad(batch)
        batch = self._numericalize_batch(batch)
        if self.include_lengths:
            batch, lengths = batch
            if to_tensor:
                lengths = torch.tensor(lengths, dtype=torch.long)

        batch = self.postprocessing(batch)
        if to_tensor:
            batch = torch.tensor(batch, dtype=torch.long)
        if self.include_lengths:
            return batch, lengths
        return batch

    def _pad(self, batch: List[List[str]]) -> List[List[str]]:
        """Pad a list of examples. The behaviour will be decided by
            the `pad_first` and `truncate_first` attribute. And of course
            the special tokens will inserted accordingly if needed.

        Args:
            batch: A list of examples, where each example is a list of tokens.

        Returns:
            List[List[str]]: The batch of padded examples. If `include_lengths`
                is True, the length will be returned together in a tuple.

        """
        if not self.is_sequential:
            return batch

        if self.fix_length is None:
            max_len = max(len(x) for x in batch)
        else:
            max_len = self.fix_length

        # start padding
        padded, lengths = [], []
        for x in batch:
            padded.append(
                _pad(
                    x,
                    max_len,
                    self.pad_token,
                    self.sos_token,
                    self.eos_token,
                    self.pad_first,
                    self.truncate_first,
                )
            )
            # (words + padding + sos + eos) - (padding)
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return (padded, lengths)
        return padded

    def _numericalize_batch(
        self,
        batch: Union[List[List[str]], Tuple[List[List[str]], List[List[int]]]],
    ) -> List[List[int]]:
        """Convert a List of examples for str to word index.

        Args:
            batch: A list of examples, where each example is a list of tokens.
                if `include_lengths` is True, it will be Tuple(examples,
                lengths).

        Returns:
            List[List[str]]: The batch of padded examples. If `include_lengths`
                is True, the length will be returned together in a tuple.

        """
        if not self.to_numericalize:
            return batch
        if not self.encoder:
            raise ValueError(
                "Encoder is not available, please `build_vector` first"
            )
        if isinstance(batch, tuple):
            batch, lengths = batch

        if self.is_target:
            # assume batch is a List of str(labels)
            batch = [[self.encoder.stoi[example]] for example in batch]
        else:
            batch = [
                [
                    self.encoder.stoi.get(token, self.encoder.unk_id)
                    for token in example
                ]
                for example in batch
            ]

        if self.include_lengths:
            return batch, lengths
        return batch

    def build_vocab(
        self,
        encoder: Union[LabelEncoder, TextEncoder],
        *datasets: List["Dataset"],
    ):
        """Build the word to index table.

        Args:
            encoder: The encoder object to build the index table.
            datasets: The attributes correspond to this `Field` of those
                Dataset` object(s) will be used to build the index table.

        Returns:
            Encoder: The fitted `Encoder`  object. Notice that the `Encoder`
                object will be changed in place after fitting.

        """
        self.encoder = encoder
        examples = []
        for dataset in datasets:
            for name, field in dataset.fields.items():
                if field == self:
                    examples += list(getattr(dataset, name))
        encoder.fit(examples)
        return encoder

    def __eq__(self, obj):
        #  return self.__dict__ == obj.__dict__
        exclude_fields = ["cleaning", "preprocessing", "postprocessing"]
        return {
            k: v for k, v in self.__dict__.items() if k not in exclude_fields
        } == {k: v for k, v in obj.__dict__.items() if k not in exclude_fields}
