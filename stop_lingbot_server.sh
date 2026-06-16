#!/usr/bin/env bash
set -euo pipefail

PID_FILE="${1:-}"
if [[ -z "${PID_FILE}" ]]; then
  echo "Usage: bash stop_lingbot_server.sh /path/to/server.pid"
  exit 2
fi
if [[ ! -f "${PID_FILE}" ]]; then
  echo "PID file not found: ${PID_FILE}"
  exit 0
fi
PID="$(cat "${PID_FILE}")"
if [[ -z "${PID}" ]]; then
  echo "PID file is empty: ${PID_FILE}"
  exit 0
fi
if kill -0 "${PID}" >/dev/null 2>&1; then
  echo "Stopping LingBot server PID ${PID}"
  kill "${PID}"
  sleep 3
  if kill -0 "${PID}" >/dev/null 2>&1; then
    echo "Process still alive; sending TERM again."
    kill "${PID}" || true
  fi
else
  echo "No running process for PID ${PID}"
fi
