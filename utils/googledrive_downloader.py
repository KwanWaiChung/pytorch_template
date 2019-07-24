from .logger import getlogger
from tqdm import tqdm
import zipfile
import requests
import os

CHUNK_SIZE = 32768
DOWNLOAD_URL = "https://docs.google.com/uc?export=download"


def download_from_googledrive(file_id: str, dst_path: str, unzip: bool = True):
    logger = getlogger(__name__)
    dst_dir = os.path.dirname(dst_path)
    if not os.path.exists(dst_dir):
        logger.info("Creating directory %s", dst_dir)
        os.makedirs(dst_dir)

    logger.info("Downloading %s...", dst_path)
    session = requests.Session()

    response = session.get(DOWNLOAD_URL, params={"id": file_id}, stream=True)

    # If file size is too big, the page will display a warming
    # The way to crack it is to get the cookie value and
    # return it in params
    token = _get_confirm_token(response)
    if token:
        response = session.get(
            DOWNLOAD_URL, params={"id": file_id, "confirm": token}, stream=True
        )

    _save_response_content(response, dst_path)

    if unzip:
        logger.info(f"Unzipping {dst_path}...")
        with zipfile.ZipFile(dst_path, "r") as z:
            z.extractall(dst_dir)
        os.remove(dst_path)
        logger.info(f"Finished unzipping.")


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _save_response_content(response, destination):
    filename = os.path.split(destination)[-1]
    size = 0
    with open(destination, "wb") as f:
        pbar = tqdm(response.iter_content(CHUNK_SIZE))
        for chunk in pbar:
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                size += CHUNK_SIZE
                pbar.set_description(
                    f"Dowloading {filename}({_sizeof_fmt(size)})"
                )


def _sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "{:.1f} {}{}".format(num, unit, suffix)
        num /= 1024.0
    return "{:.1f} {}{}".format(num, "Yi", suffix)
