from typing import Callable, List, Union, Tuple
from collections import namedtuple


class Pipeline:
    """Defines a data processing pipeline.

    Examples:
        >>> def lower(s: str) -> str:
                return s.lower()

        >>> def trim(s: str) -> str:
                return s[:3]
        >>> p = Pipeline([lower, trim])
        >>> p("HELLO")
        "hel"
        >>> p(["HELLO", "HEllo", "hello"])
        ["hel", "hel", "hel"]

    """

    def identity(s):
        return s

    def __init__(self, steps: List[Callable[[str], str]] = None):
        """
        Args:
            steps: List of Callables that preprocess the text.

        """
        if steps:
            self.steps = steps
        else:
            self.steps = [Pipeline.identity]

    def __call__(self, sentence: str):
        for step in self.steps:
            sentence = step(sentence)
        return sentence

    def add(self, step: Callable[[str], str]) -> "Pipeline":
        self.steps.append(step)
        return self
