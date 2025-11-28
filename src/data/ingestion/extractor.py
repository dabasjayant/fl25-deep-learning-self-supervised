import zipfile
from pathlib import Path

class ZipExtractor:
    def __init__(self, remove_zip: bool = False):
        self.remove_zip = remove_zip

    def extract(self, zip_path: Path, extract_dir: Path) -> None:
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Safe extraction pattern
        with zipfile.ZipFile(zip_path, 'r') as z:
            for member in z.namelist():
                # Protect against zip-slip vulnerability
                target_path = extract_dir / Path(member).name
                with z.open(member) as src, open(target_path, 'wb') as dst:
                    dst.write(src.read())

        if self.remove_zip:
            zip_path.unlink()
