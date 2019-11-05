from typing import Callable, List, Union, Tuple
from collections import namedtuple

Transform = namedtuple("Transform", ["name", "transform"])


class Pipeline:
    def identity(s):
        return s

    def __init__(self, steps: List[Tuple[str, Callable[[str], str]]]):
        """
        Args:
            steps (List): List of (name, transform) tuples where `transform`
                is a callable that receives a str and return a str
        """
        if steps:
            self.steps = [Transform(*step) for step in steps]
        else:
            self.steps = [Transform("identity", Pipeline.identity)]

    def __call__(self, sentence: Union[str, List[str]]):
        for step in self.steps:
            if isinstance(sentence, list):
                sentence = [step.transform(tok) for tok in sentence]
            else:
                sentence = step.transform(sentence)
        return sentence

    def add(
        self, steps: Union["Pipeline", Tuple[str, Callable[[str], str]]]
    ) -> "Pipeline":
        if isinstance(steps, tuple):
            self.steps.append(Transform(*steps))
        else:
            self.steps += steps.steps
        return self
