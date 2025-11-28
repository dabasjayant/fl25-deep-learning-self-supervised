import argparse

from pathlib import Path
from configs.dataset import DATA_PARTS, CHECKSUMS
from src.data.ingestion.downloader import DatasetDownloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-supervised Learning')
    parser.add_argument(
        '--download', '-d',
        action='store_true',
        help='Download datset from hugging face.'
    )
    args = parser.parse_args()

    if args.download:
        parts = DATA_PARTS
        checksums = CHECKSUMS

        downloader = DatasetDownloader(
            repo_id='tsbpp/fall2025_deeplearning',
            parts=parts,
            target_root=Path('dataset'),
            checksums=checksums,
            remove_zip_after_extract=True
        )

        downloader.download_and_prepare()
