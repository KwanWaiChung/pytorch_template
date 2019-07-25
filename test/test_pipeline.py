from ..data.pipeline import Pipeline


def test_pipeline_creation():
    def func1(s: str) -> str:
        return s.lower()

    def func2(s: str) -> str:
        return s[:3]

    p = Pipeline([("lower", func1), ("trim3", func2)])
    sentence = ["HELLO", "HEllo", "hello"]
    for s in p(sentence):
        assert s == "hel"
