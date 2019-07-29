from typing import Callable, List, Union, Tuple
from .pipeline import Pipeline
import torch


def naive_tokenizer(s: str) -> List[str]:
    return s.split()


class Field:
    def __init__(
        self,
        preprocessing: List[Tuple[str, Callable[[str], str]]] = None,
        postprocessing=None,
        is_target=False,
        tokenizer: Callable[[str], List[str]] = naive_tokenizer,
        lower=False,
        is_sequential: bool = True,
        to_numericalize: bool = True,
        include_lengths: bool = False,
        batch_first: bool = False,
        stopwords: List = None,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        sos_token: str = None,
        eos_token: str = None,
        pad_first: bool = False,
        truncate_first: bool = False,
    ):
        self.preprocessing = Pipeline(preprocessing)
        self.postprocessing = postprocessing
        self.is_target = is_target
        self.tokenizer = tokenizer
        self.lower = lower
        if lower:
            self.preprocessing = Pipeline([("lower", str.lower)]).add(
                self.preprocessing
            )
        self.is_sequential = is_sequential
        self.to_numericalize = to_numericalize
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.stopwords = set(stopwords)
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_first = pad_first
        self.truncate_first = truncate_first

    def preprocess(self, s: str) -> Union[str, List[str]]:
        if self.s_sequential:
            s = self.tokenize(s)  # List[str]
        if self.preprocessing:
            return self.preprocessing(s)
        else:
            return s

    def process(self, batch: List[List[str]]) -> torch.Tensor:
        """Do the padding and the numericalizing to create a torch.Tensor
        """
        batch = self.pad(batch)
        batch = self.numericalize(batch)
        return torch.tensor(batch, dtype=torch.long)
