#!/bin/sh
set -eu
pq2_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
pq2_python=${PQ2_COREML_PYTHON:-python3}
"$pq2_python" "$pq2_dir/generate.py" --rows 11264 6656 3840 7936 --tokens 2048 \
    --chunk-k 2048 \
    --output "$pq2_dir/models"
"$pq2_python" "$pq2_dir/generate.py" --rows 3328 --widths 6144 17408 --tokens 2048 \
    --chunk-k 1024 \
    --output "$pq2_dir/models"
