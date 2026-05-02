#!/bin/bash
# contest.sh — run baseline vs challenger across 24 parallel games.
#
# Usage:
#   ./contest.sh GARRISON_SIZE=8
#   ./contest.sh WARCHEST_LOOK_AHEAD=60 REINFORCEMENT_SIZE=12
#
# Each argument is passed as --const NAME=VALUE to contest.py.

GAMES=200
JOBS=24

# Build --const flags from positional args (fall back to a default if none given)
CONST_ARGS=()
for arg in "$@"; do
    CONST_ARGS+=(--const "$arg")
done

python3 /home/t/orbitwars/contest.py \
    --games "$GAMES" \
    --jobs  "$JOBS"  \
    "${CONST_ARGS[@]}"
