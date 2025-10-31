import os
import shutil
import logging
from typing import Optional, Sequence, Union
from huggingface_hub import snapshot_download

from app.core.settings import settings

logger = logging.getLogger(__name__)

def _safe_dir_name(repo_id: str) -> str:
    """Convert a repo_id like "org/model-name" to a safe folder name.

    Example: "org/model-name" -> "org--model-name"
    """
    return repo_id.strip().replace("/", "--")

def download_hf_model(
    repo_id: str,
    *,
    local_dir: Optional[str] = None,
    subfolder: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    local_files_only: Optional[bool] = False,
    allow_patterns: Optional[Union[str, Sequence[str]]] = None,
    ignore_patterns: Optional[Union[str, Sequence[str]]] = None,
    force_re_download: bool = False,
) -> str:
    """Download a Hugging Face model snapshot to a local directory.

    Parameters
    - repo_id: Hugging Face repository id (e.g., "org/model").
    - local_dir: Target directory to place resolved files. Defaults to `app/weights/<org--model>`.
    - subfolder: If set, only download files under this subfolder.
    - revision: Optional branch/tag/commit hash. If omitted, uses the repo default.
    - token: HF access token. If omitted, uses `settings.HF_TOKEN`.
    - local_files_only: If True, do not attempt network access. Defaults to False.
    - allow_patterns / ignore_patterns: Passed to `snapshot_download` for filtering.
    - force_re_download: If True, delete any existing local copy and re-download from HF Hub.

    Returns
    - Path to the local directory containing the snapshot files.
    """

    resolved_token = token if token is not None else settings.HF_TOKEN
    if resolved_token is not None:
        resolved_token = resolved_token.strip()
        if not resolved_token:
            resolved_token = None

    if local_dir is None:
        app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        weights_root = os.path.join(app_root, "weights")
        os.makedirs(weights_root, exist_ok=True)
        local_dir = os.path.join(weights_root, _safe_dir_name(repo_id))

    if force_re_download and os.path.exists(local_dir):
        logger.info("Force re-download enabled. Removing existing directory: %s", local_dir)
        shutil.rmtree(local_dir)

    os.makedirs(local_dir, exist_ok=True)

    if not force_re_download and os.path.exists(local_dir) and os.listdir(local_dir):
        logger.info("Model already exists locally at %s. Skipping download.", local_dir)
        final_path = os.path.join(local_dir, subfolder) if subfolder else local_dir
        return final_path

    # Build allow patterns if subfolder is specified and user didn't override
    effective_allow_patterns = allow_patterns
    if subfolder and allow_patterns is None:
        # Download everything under the subfolder
        effective_allow_patterns = [f"{subfolder}/*", subfolder]

    logger.info(
        "Downloading HF model: repo_id=%s, revision=%s, local_dir=%s, subfolder=%s, local_files_only=%s, force_re_download=%s",
        repo_id,
        revision,
        local_dir,
        subfolder,
        local_files_only,
        force_re_download,
    )

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            local_dir=local_dir,
            local_files_only=local_files_only,
            allow_patterns=effective_allow_patterns,
            ignore_patterns=ignore_patterns,
            token=resolved_token,
        )
    except Exception as e:
        logger.error("Failed to download model '%s': %s", repo_id, e)
        raise

    # If a subfolder is specified, return the subfolder path inside local_dir
    final_path = os.path.join(local_dir, subfolder) if subfolder else local_dir

    logger.info("Model available at: %s", final_path)
    return final_path


__all__ = ["download_hf_model"]
