#!/usr/bin/env bash
set -euo pipefail

# Render's native Python image includes git but not Git LFS. Install the
# upstream Linux binary into the build user's home directory.
LFS_VERSION="$(curl --fail --silent --show-error https://api.github.com/repos/git-lfs/git-lfs/releases/latest \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')"
LFS_ARCHIVE="git-lfs-linux-amd64-${LFS_VERSION}.tar.gz"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

curl --fail --location --silent --show-error \
  "https://github.com/git-lfs/git-lfs/releases/download/${LFS_VERSION}/${LFS_ARCHIVE}" \
  --output "${TMP_DIR}/${LFS_ARCHIVE}"
tar -xzf "${TMP_DIR}/${LFS_ARCHIVE}" -C "$TMP_DIR"
mkdir -p "$HOME/.local/bin"
install "${TMP_DIR}/git-lfs" "$HOME/.local/bin/git-lfs"
export PATH="$HOME/.local/bin:$PATH"

git lfs install --local
git lfs pull --include="backend/model/hvsmr_unet_split.pth"

# Stop the deployment if Git LFS left its small pointer text instead of the
# actual model weights. The expected checkpoint is about 124 MB.
python3 - <<'PY'
from pathlib import Path

checkpoint = Path("backend/model/hvsmr_unet_split.pth")
if not checkpoint.is_file():
    raise SystemExit(f"Git LFS did not provide the model checkpoint: {checkpoint}")

with checkpoint.open("rb") as model_file:
    header = model_file.read(128)

if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
    raise SystemExit("Git LFS left a pointer file; model weights were not downloaded")

if checkpoint.stat().st_size < 100_000_000:
    raise SystemExit(f"Model checkpoint is unexpectedly small: {checkpoint.stat().st_size} bytes")

print(f"Verified model weights: {checkpoint} ({checkpoint.stat().st_size} bytes)")
PY

python -m pip install -r backend/requirements.txt
