from ..data.pipeline import Pipeline


def test_pipeline():
    def func(s: str) -> str:
        return s.lower()

    p = Pipeline(func)
    sentence = ["HELLO", "HEllo", "hello"]
    for s in p(sentence):
        assert s == "hello"
