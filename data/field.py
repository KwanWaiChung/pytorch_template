from typing import Callable, List, Union


class Field:
    def __init__(
        self,
        preprocessing: List[Callable[List[str], List[str]]] = None,
        postprocessing=None,
        is_target=False,
        lower=False,
        tokenize: Callable[[str], List[str]] = None,
        sequential: bool = True,
        numericalize: bool = True,
    ):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.is_target = is_target
        self.lower = lower
        self.tokenize = tokenize
        self.sequential = sequential
        self.numericalize = numericalize

    def preprocess(self, s: str) -> Union[str, List[str]]:
        if self.sequential:
            s = self.tokenize(s)  # List[str]
        if self.processing:
            return self.processing(s)
        else:
            return s
