from torch.utils.data import DataLoader
from allennlp.data import Vocabulary, PyTorchDataLoader
from ....datasets.text_classification import TextClassificationDataset
import torch
import random
import os
import json

# 170 -> 101010, good balance of 1 and 0
torch.manual_seed(170)
random.seed(170)


class YelpDataset(TextClassificationDataset):
    def __init__(self):
        super().__init__(
            max_len=50, label_transform=[lambda x: "pos" if x > 3 else "neg"]
        )

    def _read(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        for i, row in enumerate(data):
            if i >= 30:
                return
            yield self.text_to_instance(row["text"], row["stars"])


def getYelpDataloader():
    reader = YelpDataset()
    train_dataset = reader.read(getJsonFilename())
    train_dataset, val_dataset = reader.split(train_dataset, 0.8)
    vocab = Vocabulary.from_instances(train_dataset)
    train_dataset.index_with(vocab)
    val_dataset.index_with(vocab)
    return (
        PyTorchDataLoader(train_dataset, shuffle=True, batch_size=4),
        PyTorchDataLoader(val_dataset, batch_size=1),
        vocab.get_vocab_size("tokens"),
    )


def getLinearDataloader(input_size):
    train_X = torch.rand(80, input_size)
    train_y = (100 * train_X).sum(dim=1, keepdim=True) + 1

    val_X = torch.rand(80, input_size)
    val_y = (100 * val_X).sum(dim=1, keepdim=True) + 1 + torch.randn(80, 1)

    train_X = train_X.reshape(-1, 4, input_size)
    train_y = train_y.reshape(-1, 4)
    val_X = val_X.reshape(-1, 4, input_size)
    val_y = val_y.reshape(-1, 4)

    return [
        {"train_X": train_X[i], "train_y": train_y[i]}
        for i in range(len(train_X))
    ], [{"val_X": val_X[i], "val_y": val_y[i]} for i in range(len(val_X))]


def getTextData(n=50):
    """An example text file with no label

    Returns:
        List of str.
    """
    with open(
        f"{os.path.dirname(os.path.realpath(__file__))}"
        "/../../data/test_data.txt",
        "r",
        encoding="utf-8",
    ) as f:
        data = random.choices([l.strip() for l in f], k=n)
    return data


def getCsvFilename():
    return (
        f"{os.path.dirname(os.path.realpath(__file__))}/../../data/train.csv"
    )


def getJsonFilename():
    """
    Yelp
    """
    return (
        f"{os.path.dirname(os.path.realpath(__file__))}"
        "/../../data/reviews.json"
    )
