from ..datasets.text_classification import ImdbDataset
import os
import shutil


def test_dataset_download():
    reader = ImdbDataset(file_path="test_data")
    assert "train-pos.txt" in os.listdir("test_data")


def test_dataset_read():
    reader = ImdbDataset(file_path="test_data")
    dataset = reader.read("train")
    assert len(dataset) == 25000
    assert "text" in dataset[0].fields
    assert "label" in dataset[0].fields


def test_dataset_read_n_examples():
    reader = ImdbDataset(file_path="test_data", n_examples=20)
    dataset = reader.read("train")
    assert len(dataset) == 20


def test_dataset_text_transform():
    reader = ImdbDataset(
        file_path="test_data",
        text_transform=[lambda x: x.upper(), lambda x: x[:5]],
        n_examples=20,
    )
    dataset = reader.read("train")
    for s in dataset[0]["text"].tokens:
        assert s.text.isupper()
    assert len(dataset[0]["text"].tokens) <= 5


def test_dataset_max_len():
    reader = ImdbDataset(
        file_path="test_data",
        n_examples=20,
        max_len=20,
    )
    dataset = reader.read("train")
    for instance in dataset:
        assert len(instance["text"].tokens) <= 20


def test_dataset_label_transform():
    reader = ImdbDataset(
        file_path="test_data",
        label_transform=[lambda x: x.upper()],
        n_examples=20,
    )
    dataset = reader.read("train")
    assert dataset[0]["label"].label.isupper()
    shutil.rmtree("test_data")
