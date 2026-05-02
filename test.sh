#!/usr/bin/env bash
set -euo pipefail

python3 hellburner_tests.py "$@"
python3 hellburner_integration-1.py "$@"
python3 hellburner_integration-2.py "$@"
