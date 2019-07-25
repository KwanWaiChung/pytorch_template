from typing import Callable, List, Union


class Pipeline:
    def __init__(self, name: str, func: Callable[[str], str]):
        if func:
            self.func = func
        else:
            self.func = Pipeline.identity
        self.next = None
        self.name = name

    def __call__(self, sentence: List[str]):
        x = self.func(sentence)
        if self.next:
            return self.next(x)
        return x

    def add(
        self, pipeline: Union["Pipeline", Callable[[str], str]]
    ) -> "Pipeline":
        if not isinstance(pipeline, Pipeline):
            pipeline = Pipeline(pipeline)
        pipeline.next = self.next
        self.next = pipeline
        return self

    @staticmethod
    def identity(s: str) -> str:
        return s
