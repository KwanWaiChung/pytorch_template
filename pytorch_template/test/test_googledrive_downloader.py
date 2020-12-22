import pytest
import os
import shutil
from ..utils.googledrive_downloader import download_from_googledrive


path = "test_data"


def test_download_small_zip():
    _id = "1RmAPdqCG69UFnhL8--PPzi7TlcNcBV1C"
    download_from_googledrive(_id, path, True)
    assert "test-neg.txt" in os.listdir(path)
    assert "test-pos.txt" in os.listdir(path)
    assert "train-neg.txt" in os.listdir(path)
    assert "train-pos.txt" in os.listdir(path)
    shutil.rmtree(path)


def test_download_large_zip():
    _id = "1jAicm2f8OqCe28No3qo7pPaXBHEP7RuJ"
    download_from_googledrive(_id, path, True)
    assert "train.csv" in os.listdir(path)
    assert "test.csv" in os.listdir(path)
    assert "test_labels.csv" in os.listdir(path)
    assert "sample_submission.csv" in os.listdir(path)
    shutil.rmtree(path)
