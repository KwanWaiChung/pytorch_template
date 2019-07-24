from typing import Callable, List, Union


class Pipeline:
    def __init__(self, func: Callable[List[str], List[str]]):
        if func is None:
            self.func = func
        else:
            self.func = Pipeline.identity
        self.pipes[self]

    def __call__(self, sentence: List[str]):
        for pipe in self.pipes:
            sentence = pipe(sentence)
        return sentence

    def add_before(
        self, pipeline: Union["Pipeline", Callable[List[str], List[str]]]
    ) -> "Pipeline":
        if not isinstance(pipeline, Pipeline):
            pipeline = Pipeline(pipeline)
        self.pipes = pipeline.pipes + self.pipes
        return self

    def add_after(
        self, pipeline: Union["Pipeline", Callable[List[str], List[str]]]
    ) -> "Pipeline":
        if not isinstance(pipeline, Pipeline):
            pipeline = Pipeline(pipeline)
        self.pipes = self.pipes + pipeline.pipes
        return self

    @staticmethod
    def identity(sentence: List[str]):
        return sentence
