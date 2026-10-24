#!/usr/bin/env bash
set -euo pipefail

cd /teamspace/studios/this_studio/cloud_removal_india
mkdir -p data/raw data/patches data/splits models results logs reports
python -m pip install --upgrade pip
pip install -r requirements.txt
python - <<'PY'
import numpy
print(f"Verified numpy={numpy.__version__}")
major = int(numpy.__version__.split(".")[0])
if major >= 2:
    raise SystemExit("numpy>=2 is incompatible with this Lightning stack. Setup failed intentionally.")
PY
echo "Setup complete."
