from ..data.field import Field


def func1(s: str) -> str:
    return s.lower()


def func2(s: str) -> str:
    return s[:3]


def test_field_preprocess():
    transform = [("lower", func1), ("trim3", func2)]

    field = Field(preprocessing=transform, sequential=False)
    assert field.preprocess("HELLO") == "hel"


def test_field_lower():
    transform = [("trim3", func2)]
    field = Field(preprocessing=transform, lower=True, sequential=False)
    assert (
        field.preprocess("HELLO") == "hel"
    ), "parameter `lower` is not working"


def test_field_sequential():
    transform = [("lower", func1), ("trim3", func2)]
    field = Field(preprocessing=transform, sequential=True)
    assert field.preprocess("HELLO WOrld") == ["hel", "wor"]
