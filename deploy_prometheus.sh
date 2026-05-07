#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/sem2-mlops-assignment-1}"
PROM_SRC="${REPO_DIR}/prometheus.yml"
PROM_DST="/etc/prometheus/prometheus.yml"

echo "[INFO] Using repo: ${REPO_DIR}"
cd "${REPO_DIR}"

echo "[INFO] Pulling latest code..."
git pull

if [[ ! -f "${PROM_SRC}" ]]; then
  echo "[ERROR] Missing ${PROM_SRC}"
  exit 1
fi

echo "[INFO] Copying ${PROM_SRC} -> ${PROM_DST}"
sudo cp "${PROM_SRC}" "${PROM_DST}"

echo "[INFO] Restarting Prometheus service..."
sudo systemctl restart prometheus
sudo systemctl status prometheus --no-pager

echo "[INFO] Prometheus config deployed successfully."
