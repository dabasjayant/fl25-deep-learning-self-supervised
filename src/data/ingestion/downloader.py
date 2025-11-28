# src/data/ingestion/dataset_downloader.py

from pathlib import Path
from typing import List
from .hf_client import HuggingFaceRepoClient
from .checksum import ChecksumValidator
from .extractor import ZipExtractor

class DatasetDownloader:
    def __init__(
        self,
        repo_id: str,
        parts: List[str],
        target_root: Path,
        checksums: dict,
        revision: str = 'main',
        remove_zip_after_extract: bool = False
    ):
        self.repo = HuggingFaceRepoClient(repo_id=repo_id, revision=revision)
        self.parts = parts
        self.target_root = target_root
        self.checksums = checksums

        self.download_dir = target_root / 'zip_files'
        self.extract_dir = target_root / 'images'

        self.validator = ChecksumValidator(checksums)
        self.extractor = ZipExtractor(remove_zip=remove_zip_after_extract)

    def download_and_prepare(self):
        print('Listing files in repo...')
        repo_files = self.repo.list_files()

        for fname in self.parts:
            if fname not in repo_files:
                raise FileNotFoundError(f'{fname} not found in HF repo.')

            print(f'Downloading {fname}...')
            zip_path = self.repo.download_file(fname, dest_dir=self.download_dir)

            print(f'Validating checksum for {fname}...')
            self.validator.validate(zip_path)

            print(f'Extracting {fname}...')
            self.extractor.extract(zip_path, self.extract_dir)

        print('All dataset parts processed successfully.')
