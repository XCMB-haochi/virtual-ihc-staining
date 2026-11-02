from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ihc_dinov2 import (
    DINOV2_REPOSITORY, DINOV2_REVISION, DINOV2_WEIGHT_PATH,
    DINOV2_WEIGHT_SHA256, DINOV2_WEIGHT_URL,
)


def main() -> None:
    if not DINOV2_REPOSITORY.exists():
        DINOV2_REPOSITORY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none",
                        "https://github.com/facebookresearch/dinov2.git", str(DINOV2_REPOSITORY)], check=True)
        subprocess.run(["git", "-C", str(DINOV2_REPOSITORY), "checkout", "--detach", DINOV2_REVISION], check=True)
    revision = subprocess.check_output(["git", "-C", str(DINOV2_REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
    if revision != DINOV2_REVISION:
        raise ValueError("Existing DINOv2 source has another revision; it was left untouched.")
    if not DINOV2_WEIGHT_PATH.exists():
        DINOV2_WEIGHT_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = DINOV2_WEIGHT_PATH.with_suffix(".download")
        urllib.request.urlretrieve(DINOV2_WEIGHT_URL, temporary)
        with temporary.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != DINOV2_WEIGHT_SHA256:
                raise ValueError("DINOv2 download hash mismatch.")
        temporary.replace(DINOV2_WEIGHT_PATH)
    with DINOV2_WEIGHT_PATH.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != DINOV2_WEIGHT_SHA256:
            raise ValueError("Existing DINOv2 weight hash mismatch.")
    print("DINOv2 source and weights ready:", DINOV2_REVISION, flush=True)


if __name__ == "__main__":
    main()
