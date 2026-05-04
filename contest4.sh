#!/bin/bash
# contest4.sh — run baseline×3 vs challenger across 24 parallel 4-way games.
#
# Usage:
#   ./contest4.sh GARRISON_SIZE=8
#   ./contest4.sh WARCHEST_LOOK_AHEAD=60 REINFORCEMENT_SIZE=12
#
# Each argument is passed as --const NAME=VALUE to contest4.py.

GAMES=200
JOBS=24

# Build --const flags from positional args (fall back to a default if none given)
CONST_ARGS=()
for arg in "$@"; do
    CONST_ARGS+=(--const "$arg")
done

python3 /home/t/orbitwars/contest4.py \
    --games "$GAMES" \
    --jobs  "$JOBS"  \
    "${CONST_ARGS[@]}"
