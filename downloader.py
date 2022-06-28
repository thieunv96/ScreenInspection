import gdown
import py7zr

import os
import zipfile


def download(url, output):
    print("[INFO] download the dataset...")
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    with py7zr.SevenZipFile(output, mode='r') as z:
        z.extractall()


if __name__ == "__main__":
    url = 'https://drive.google.com/u/0/uc?id=1hKLC6rHtG9yv5mcFc_5Q-5-10Wo1zP8o&export=download'
    output = 'dataset.7z'
    download(url, output)