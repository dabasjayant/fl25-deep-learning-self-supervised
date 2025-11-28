import hashlib
from pathlib import Path
from typing import Dict

class ChecksumValidator:
    def __init__(self, checksums: Dict[str, str]):
        """
        checksums: dict mapping filename -> expected sha256
        """
        self.checksums = checksums

    @staticmethod
    def sha256_of_file(filepath: Path) -> str:
        hash_sha = hashlib.sha256()
        with filepath.open('rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_sha.update(chunk)
        return hash_sha.hexdigest()

    def validate(self, filepath: Path) -> bool:
        fname = filepath.name
        if fname not in self.checksums:
            raise ValueError(f'No checksum provided for {fname}')

        expected = self.checksums[fname]
        actual = self.sha256_of_file(filepath)

        if actual != expected:
            raise ValueError(f'Checksum mismatch for {fname}. Expected {expected}, got {actual}')

        return True
