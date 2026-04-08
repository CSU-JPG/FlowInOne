#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

python "$SCRIPT_DIR/unified_to_tars.py" \
  --root /path/to/your_source_root \
  --tar-dir /path/to/your_tar_output \
  --samples-per-shard 600 \
  # --key-prefix None \
  # --data-type t2i \
  --read-workers 32