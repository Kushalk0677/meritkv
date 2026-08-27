#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${MERITKV_COLAB_SESSION:-meritkv-native-bypass-$(date +%s)}"
COLAB_AUTH="${COLAB_AUTH:-oauth2}"
KEEP_SESSION="${KEEP_SESSION:-0}"
REMOTE_DIR="/content/meritkv_native_bypass"
REMOTE_ARCHIVE="/content/meritkv_native_vllm_execute_bypass_results.zip"
LOCAL_RESULTS_DIR="${MERITKV_RESULTS_DIR:-${PACKAGE_DIR}/downloaded_results}"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_ARCHIVE="${LOCAL_RESULTS_DIR}/meritkv_native_vllm_execute_bypass_results_${RUN_TAG}.zip"
SESSION_STARTED=0
SOURCE_ARCHIVE=""

cleanup() {
    local status=$?
    if [[ -n "${SOURCE_ARCHIVE}" && -f "${SOURCE_ARCHIVE}" ]]; then
        rm -f -- "${SOURCE_ARCHIVE}"
    fi
    if [[ "${SESSION_STARTED}" == "1" && "${KEEP_SESSION}" != "1" ]]; then
        "${COLAB_CMD[@]}" stop -s "${SESSION}" >/dev/null 2>&1 || true
    fi
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$(uname -s)" != "Linux" && "$(uname -s)" != "Darwin" ]]; then
    echo "Run this workflow inside WSL2 on Windows."
    exit 1
fi

if command -v colab >/dev/null 2>&1; then
    COLAB_BIN="$(command -v colab)"
elif command -v uv >/dev/null 2>&1; then
    uv tool install google-colab-cli
    COLAB_BIN="$(uv tool dir --bin)/colab"
else
    CLI_VENV="${XDG_DATA_HOME:-${HOME}/.local/share}/meritkv-colab-cli-venv"
    python3 -m venv "${CLI_VENV}"
    "${CLI_VENV}/bin/python" -m pip install --upgrade pip google-colab-cli
    COLAB_BIN="${CLI_VENV}/bin/colab"
fi

COLAB_CMD=("${COLAB_BIN}" "--auth=${COLAB_AUTH}")
"${COLAB_CMD[@]}" version

echo "Provisioning Colab T4 session ${SESSION}..."
SESSION_STARTED=1
"${COLAB_CMD[@]}" new -s "${SESSION}" --gpu T4
"${COLAB_CMD[@]}" status -s "${SESSION}"
printf '%s\n' "from pathlib import Path; Path('${REMOTE_DIR}').mkdir(parents=True, exist_ok=True)" | "${COLAB_CMD[@]}" exec -s "${SESSION}"

for file in run_native_execute_bypass.py run_experiment.sh run_remote_experiment.py finalize_manifest.py requirements.txt; do
    "${COLAB_CMD[@]}" upload -s "${SESSION}" "${PACKAGE_DIR}/${file}" "${REMOTE_DIR}/${file}"
done

SOURCE_ARCHIVE="$(mktemp --suffix=.zip)"
python3 "${PACKAGE_DIR}/package_source_snapshot.py" "${PACKAGE_DIR}/../../.." "${SOURCE_ARCHIVE}"
"${COLAB_CMD[@]}" upload -s "${SESSION}" "${SOURCE_ARCHIVE}" "${REMOTE_DIR}/meritkv_source.zip"
rm -f -- "${SOURCE_ARCHIVE}"
SOURCE_ARCHIVE=""

set +e
"${COLAB_CMD[@]}" exec -s "${SESSION}" -f "${PACKAGE_DIR}/run_remote_experiment.py"
REMOTE_STATUS=$?
set -e

mkdir -p -- "${LOCAL_RESULTS_DIR}"
"${COLAB_CMD[@]}" download -s "${SESSION}" "${REMOTE_ARCHIVE}" "${LOCAL_ARCHIVE}"
sha256sum "${LOCAL_ARCHIVE}"
echo "Downloaded: ${LOCAL_ARCHIVE}"
set +e
python3 "${PACKAGE_DIR}/verify_result_archive.py" "${LOCAL_ARCHIVE}"
ARCHIVE_STATUS=$?
set -e
if [[ "${REMOTE_STATUS}" != "0" ]]; then
    echo "The remote driver returned ${REMOTE_STATUS}; the downloaded archive contains retained failure evidence."
fi
if [[ "${ARCHIVE_STATUS}" != "0" ]]; then
    echo "The downloaded archive did not pass run-status verification."
fi
if [[ "${REMOTE_STATUS}" != "0" ]]; then
    exit "${REMOTE_STATUS}"
fi
exit "${ARCHIVE_STATUS}"
