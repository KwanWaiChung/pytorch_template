import pytest
import os
import shutil
from ..utils.googledrive_downloader import download_from_googledrive


def test_download_small_zip():
    _id = "1ZdLv2roUU5HiNKMvWcaviCiNPG1knLBr"
    curr_path = os.path.dirname(os.path.realpath(__file__))
    download_from_googledrive(_id, os.path.join(curr_path, "data.zip"), True)
    assert "ToxicDataset" in os.listdir(curr_path)
    shutil.rmtree(os.path.join(curr_path, "ToxicDataset"))


def test_download_large_zip():
    _id = "18cNTr5kLfCj4-LudLBvR2ch8m3ThgOEs"
    curr_path = os.path.dirname(os.path.realpath(__file__))
    download_from_googledrive(_id, os.path.join(curr_path, "data.zip"), True)
    assert "train.csv" in os.listdir(curr_path)
    os.remove(os.path.join(curr_path, "train.csv"))
    os.remove(os.path.join(curr_path, "test.csv"))
    os.remove(os.path.join(curr_path, "val.csv"))
