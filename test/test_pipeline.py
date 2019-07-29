from ..data.pipeline import Pipeline


def get_pipeline():
    def func1(s: str) -> str:
        return s.lower()

    def func2(s: str) -> str:
        return s[:3]

    p = Pipeline([("lower", func1), ("trim3", func2)])
    return p


def test_pipeline_process_list():
    p = get_pipeline()
    sentence = ["HELLO", "HEllo", "hello"]
    for s in p(sentence):
        assert s == "hel"


def test_pipeline_process_str():
    p = get_pipeline()
    sentence = "HELLO"
    assert p(sentence) == "hel"


def test_pipeline_add_list():
    p = get_pipeline()

    def func1(s: str) -> str:
        return s.upper()

    def func2(s: str) -> str:
        return s + "_success"

    p = p.add(("upper", func1)).add(("add_words", func2))
    assert p.steps[2].name == "upper"
    assert p.steps[3].name == "add_words"
    sentence = ["HELLO", "HEllo", "hello"]
    for s in p(sentence):
        assert s == "HEL_success"


def test_pipeline_add_pipeine():
    def func1(s: str) -> str:
        return s.upper()

    def func2(s: str) -> str:
        return s + "_success"

    p = get_pipeline()
    p2 = Pipeline([("upper", func1), ("add_words", func2)])
    p.add(p2)
    assert p.steps[2].name == "upper"
    assert p.steps[3].name == "add_words"
    sentence = ["HELLO", "HEllo", "hello"]
    for s in p(sentence):
        assert s == "HEL_success"
