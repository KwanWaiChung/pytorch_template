from collections import Counter
from typing import List
from .misc import flatten_2d
import numpy as np
from .logger import getlogger


class LabelEncoder:
    def __init__(self, itos=None, stoi=None, stof=None):
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

    def fit(self, y, special_tokens=[]):
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
        y = flatten_2d(y)
        return [self.stoi[token] for token in y]

    def decode(self, y):
        return [self.itos[idx] for idx in y]


class TextEncoder(LabelEncoder):
    def __init__(
        self,
        pad_token="<pad>",
        unk_token="<unk>",
        sos_token=None,
        eos_token=None,
    ):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token

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

    def fit(self, y):
        super().fit(
            y,
            special_tokens=[
                self.unk_token,
                self.pad_token,
                self.sos_token,
                self.eos_token,
            ],
        )
