from collections import Counter
from typing import List, Dict
from .misc import flatten_2d
from .logger import getlogger
from ..exceptions import NotFittedError
import pickle


class LabelEncoder:
    """An object that does the word to index

    Attributes:
        itos (List[str]): A list that maps index to label.
        stoi (Dict[str, int]): A dict that maps label to index.
        stof (Dict[str, int]): A dict that maps label to frequency.
    """

    def __init__(
        self,
        itos: List[str] = None,
        stoi: Dict[str, int] = None,
        stof: Dict[str, int] = None,
    ):
        self.itos = itos
        self.stoi = stoi
        self.stof = stof
        if not self.itos:
            self.itos = []
        if not self.stoi:
            self.stoi = {}
        if not self.stof:
            self.stof = {}
        self.fitted = False
        self.logger = getlogger(__name__)

    def fit(
        self, y: List[str], special_tokens: List[str] = []
    ) -> "LabelEncoder":
        """Build the word indexes according to corpus. If it was fitted
            before, then it will do nothing.

        Args:
            y (List[str]): The list of label.
            special_tokens(List[str]): The list of labels that will be
                excluded when encoding.
        """
        if self.fitted:
            self.logger.debug("The encoder has already been fitted")
            return self

        c = Counter(flatten_2d(y))
        self.stof.update(c)
        for word in c.keys():
            if word not in special_tokens:
                self.itos.append(word)

        for idx, word in enumerate(self.itos):
            self.stoi[word] = idx
        self.logger.debug("The word indexes has been built")
        self.fitted = True
        return self

    def encode(self, y: List[str]):
        """Encoding a list of labels into a list of indexes

        Args:
            y (List[str]): The sentence to encode.

        Returns:
            List[int]: The list of word indexes
        """
        if not self.fitted:
            raise NotFittedError("Encoder is not fitted")
        y = flatten_2d(y)
        return [self.stoi[token] for token in y]

    def decode(self, y):
        """Decoding a list of indexes into a list of labels

        Args:
            y (List[int]): The sequence to decode.

        Returns:
            List[str]: The decoded list of class labels
        """
        if not self.fitted:
            raise NotFittedError("Encoder is not fitted")
        y = flatten_2d(y)
        return [self.itos[idx] for idx in y]

    def dump(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str):
        with open(filename, "rb") as f:
            encoder = pickle.load(f)
        return encoder

    def __getstate__(self):
        return {"stof": self.stof, "itos": self.itos}

    def __setstate__(self, d):
        self.stof = d["stof"]
        self.itos = d["itos"]
        self.stoi = {word: idx for idx, word in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)


class TextEncoder(LabelEncoder):
    def __init__(
        self,
        pad_token="<pad>",
        unk_token="<unk>",
        sos_token=None,
        eos_token=None,
        max_size=float("inf"),
        min_freq=1,
    ):
        """Handles the mapping of word to index

        Args:
            pad_token: The token used for padding.
            unk_token: The token used for out of vocabulary words.
            sos_token: The token used for specifying the start of a sentence.
            eos_token: The toekn used for specifying the end of a sentence.
            max_size: The maximum size of the vocabulary.
            min_freq: The minimum frequency needed to incldue a token.

        """
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_size = max_size
        self.min_freq = min_freq

        itos = []
        if unk_token:
            self.unk_id = len(itos)
            itos.append(unk_token)
        if pad_token:
            self.pad_id = len(itos)
            itos.append(pad_token)
        if sos_token:
            self.sos_id = len(itos)
            itos.append(sos_token)
        if eos_token:
            self.eos_id = len(itos)
            itos.append(eos_token)
        super().__init__(itos=itos)

    def fit(
        self, y: List[str], special_tokens: List[str] = []
    ) -> "LabelEncoder":
        """Build the word indexes according to corpus. If it was fitted
            before, then it will do nothing.

        Args:
            y (List[str]): The list of label.
            special_tokens(List[str]): The list of labels that will be
                excluded when encoding.
        """
        if self.fitted:
            self.logger.debug("The encoder has already been fitted")
            return self

        c = Counter(flatten_2d(y))
        self.stof.update(c)
        for i, (word, freq) in enumerate(c.items()):
            if i >= self.max_size:
                break
            if word not in special_tokens and freq >= self.min_freq:
                self.itos.append(word)

        for idx, word in enumerate(self.itos):
            self.stoi[word] = idx
        self.logger.debug("The word indexes has been built")
        self.fitted = True
        return self

    def __getstate__(self):
        d = super().__getstate__()
        d["pad_token"] = self.pad_token
        d["unk_token"] = self.unk_token
        d["sos_token"] = self.sos_token
        d["eos_token"] = self.eos_token
        return d

    def __setstate__(self, d):
        super().__setstate__(d)
        self.pad_token = d.get("pad_token")
        self.unk_token = d.get("unk_token")
        self.sos_token = d.get("sos_token")
        self.eos_token = d.get("eos_token")
        if self.pad_token:
            self.pad_id = self.stoi[self.pad_token]
        if self.unk_token:
            self.unk_id = self.stoi[self.unk_token]
        if self.sos_token:
            self.sos_id = self.stoi[self.sos_token]
        if self.eos_token:
            self.eos_id = self.stoi[self.eos_token]
