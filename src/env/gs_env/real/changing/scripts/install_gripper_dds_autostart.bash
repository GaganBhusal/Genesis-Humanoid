#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo bash "$0" "$@"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SERVICE_NAME="${SERVICE_NAME:-gripper-dds.service}"
SERVICE_USER="${SUDO_USER:-${USER}}"
SERVICE_GROUP="${SERVICE_GROUP:-${SERVICE_USER}}"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}"
ENV_FILE="/etc/default/gripper-dds"
RUN_SCRIPT="${WORK_DIR}/scripts/run_gripper_dds_server.bash"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "Run script not found: ${RUN_SCRIPT}"
  exit 1
fi

cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Gripper DDS Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${WORK_DIR}
EnvironmentFile=-${ENV_FILE}
ExecStart=/usr/bin/env bash ${RUN_SCRIPT}
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

if [[ ! -f "${ENV_FILE}" ]]; then
  cat > "${ENV_FILE}" <<'EOF'
# Optional overrides for gripper DDS startup
# PYTHON_BIN=/absolute/path/to/python
# GRIPPER_PORT=/dev/ttyUSB0
# GRIPPER_DOMAIN=0
# GRIPPER_HZ=25.0
# GRIPPER_BAUDRATE=115200
EOF
fi

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"

echo "Installed and started ${SERVICE_NAME}"
echo "Check status with: systemctl status ${SERVICE_NAME}"
echo "Logs with: journalctl -u ${SERVICE_NAME} -f"
