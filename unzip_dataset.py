import zipfile, os
from pathlib import Path


DATASET_FOLDER_NAME = '.dataset' # hide uncompressed data from version control


def extract_dataset(extract_if_dataset_present=False) -> None:

    if DATASET_FOLDER_NAME in os.listdir() and not extract_if_dataset_present:
        print("found dataset folder, return from archive extraction procedure without exracting")
        print(f"rename or delete folder {DATASET_FOLDER_NAME} if you want to force extraction")
        return

    # snipppet from https://dev.to/abbazs/how-to-unzip-multiple-zip-files-in-a-folder-using-python-6dd
    p = Path('.')
    for f in p.glob('*.zip'):
        with zipfile.ZipFile(f, 'r') as archive:
            # naming convention: SpotifyFeatures.csv.zip contains SpotifyFeatures.csv
            archive.extractall(path=f'.dataset')


if __name__ == '__main__':
    extract_dataset(extract_if_dataset_present=True)
