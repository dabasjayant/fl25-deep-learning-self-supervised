from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path

class HuggingFaceRepoClient:
    def __init__(self, repo_id: str, revision: str = 'main'):
        self.repo_id = repo_id
        self.revision = revision

    def list_files(self) -> list:
        return list_repo_files(self.repo_id, revision=self.revision, repo_type='dataset')

    def download_file(self, filename: str, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        downloaded_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            revision=self.revision,
            repo_type='dataset',
            local_dir=dest_dir
        )
        return Path(downloaded_path)
