#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FOLDER_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SERVER_SCRIPT="${FOLDER_ROOT}/gripper_dds_server.py"

exec python "${SERVER_SCRIPT}"
