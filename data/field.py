from typing import Callable, List


class Field:
    def __init__(
        self,
        preprocessing: List[Callable[List[str], List[str]]] = None,
        postprocessing=None,
        is_target=False,
        lower=False,
        tokenize: Callable[[str], List[str]] = None,
    ):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.is_target = is_target
        self.lower = lower
        self.tokenize = tokenize

    def preprocess(self, s: str):
        if not self.is_target:
            s = self.tokenize(s)
            # s: List[str]
        if self.processing:
            return self.processing(s)
        else:
            return s
