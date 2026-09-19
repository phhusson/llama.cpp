#!/bin/sh
set -eu
if [ "$#" -eq 0 ]; then
    echo "usage: $0 COMMAND [ARGS...]" >&2
    exit 2
fi
export GGML_METAL_ANE=1
exec "$@"
