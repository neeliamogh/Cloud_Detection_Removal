#!/usr/bin/env bash
set -euo pipefail

cd /teamspace/studios/this_studio/cloud_removal_india
mkdir -p results
python - <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path("/teamspace/studios/this_studio/cloud_removal_india")
out = root / "results" / "cloud_removal_india_results.zip"
include_dirs = ["results", "models", "reports", "logs", "configs", "data"]

with ZipFile(out, "w", ZIP_DEFLATED) as zf:
    for dirname in include_dirs:
        base = root / dirname
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.resolve() != out.resolve():
                zf.write(path, path.relative_to(root))

print(f"Created {out}")
PY
echo "Created results/cloud_removal_india_results.zip"
