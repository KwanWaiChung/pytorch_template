from ..data.pipeline import Pipeline


def get_pipeline():
    def func1(s: str) -> str:
        return s.lower()

    def func2(s: str) -> str:
        return s[:3]

    p = Pipeline([func1, func2])
    return p


def test_pipeline_process_list():
    p = get_pipeline()
    sentence = ["HELLO", "HEllo", "hello"]
    for word in sentence:
        assert p(word) == "hel"


def test_pipeline_process_str():
    p = get_pipeline()
    sentence = "HELLO"
    assert p(sentence) == "hel"


def test_pipeline_add():
    p = get_pipeline()

    def func1(s: str) -> str:
        return s.upper()

    def func2(s: str) -> str:
        return s + "_success"

    p = p.add(func1).add(func2)
    sentence = ["HELLO", "HEllo", "hello"]
    for word in sentence:
        assert p(word) == "HEL_success"
