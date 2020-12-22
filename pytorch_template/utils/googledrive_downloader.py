from .logger import getlogger
from tqdm import tqdm
import zipfile
import requests
import os
import re

CHUNK_SIZE = 32768
DOWNLOAD_URL = "https://drive.google.com/uc?export=download"


def download_from_googledrive(
    file_id: str, dst_dir: str = "", unzip: bool = True
):
    """
    Args:
        dst_dir: The folder to store the downloaded file.
    """
    logger = getlogger(__name__)
    session = requests.Session()
    response = session.get(DOWNLOAD_URL, params={"id": file_id}, stream=True)

    #  dst_dir, filename = os.path.split(dst_path)
    if dst_dir and not os.path.exists(dst_dir):
        logger.info("Creating directory %s", dst_dir)
        os.makedirs(dst_dir)

    # If file size is too big, the page will display a warming
    # The way to crack it is to get the cookie value and
    # return it in params
    token = _get_confirm_token(response)
    if token:
        logger.info(f"Got the confirm tokem.")
        response = session.get(
            DOWNLOAD_URL, params={"id": file_id, "confirm": token}, stream=True
        )
    filename = re.search(
        r"filename\=\"(.*)\"", response.headers["Content-Disposition"]
    ).group(1)
    dst_path = os.path.join(dst_dir, filename)
    logger.info("Downloading %s...", filename)
    _save_response_content(response, dst_path)

    if unzip:
        logger.info(f"Unzipping {filename}...")
        with zipfile.ZipFile(dst_path, "r") as z:
            z.extractall(dst_dir)
        # remove the zip too
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
