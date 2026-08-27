#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${MERITKV_COLAB_SESSION:-meritkv-splice-$(date +%s)}"
COLAB_AUTH="${COLAB_AUTH:-oauth2}"
KEEP_SESSION="${KEEP_SESSION:-0}"
REMOTE_DIR="/content/meritkv_splice"
REMOTE_DRIVER="${MERITKV_REMOTE_DRIVER:-run_remote_validation.py}"
REMOTE_ARCHIVE="${MERITKV_REMOTE_ARCHIVE:-/content/meritkv_exact_splice_validation_outputs.zip}"
LOCAL_RESULTS_DIR="${MERITKV_RESULTS_DIR:-${PACKAGE_DIR}/downloaded_results}"
MODEL_SELECTION="${MERITKV_MODELS:-gemma4_e2b,qwen25_15b}"
ATTENTION_SELECTION="${MERITKV_ATTENTION:-eager,sdpa}"
MODEL_SELECTION_TAG="${MODEL_SELECTION//,/__}"
ATTENTION_SELECTION_TAG="${ATTENTION_SELECTION//,/__}"
OUTPUT_TAG="${MERITKV_OUTPUT_TAG:-${MODEL_SELECTION_TAG}_${ATTENTION_SELECTION_TAG}}"
LOCAL_ARCHIVE="${LOCAL_RESULTS_DIR}/meritkv_exact_splice_validation_outputs_${OUTPUT_TAG}.zip"
TOKEN_FILE=""
CLI_VERSION_FILE=""
MODEL_SELECTION_FILE=""
ATTENTION_SELECTION_FILE=""
SESSION_STARTED=0

cleanup() {
    local exit_code=$?
    if [[ -n "${TOKEN_FILE}" && -f "${TOKEN_FILE}" ]]; then
        rm -f -- "${TOKEN_FILE}"
    fi
    if [[ -n "${CLI_VERSION_FILE}" && -f "${CLI_VERSION_FILE}" ]]; then
        rm -f -- "${CLI_VERSION_FILE}"
    fi
    if [[ -n "${MODEL_SELECTION_FILE}" && -f "${MODEL_SELECTION_FILE}" ]]; then
        rm -f -- "${MODEL_SELECTION_FILE}"
    fi
    if [[ -n "${ATTENTION_SELECTION_FILE}" && -f "${ATTENTION_SELECTION_FILE}" ]]; then
        rm -f -- "${ATTENTION_SELECTION_FILE}"
    fi
    if [[ "${SESSION_STARTED}" == "1" ]]; then
        "${COLAB_CMD[@]}" rm -s "${SESSION}" "${REMOTE_DIR}/.hf_token" \
            >/dev/null 2>&1 || true
    fi
    if [[ "${SESSION_STARTED}" == "1" && "${KEEP_SESSION}" != "1" ]]; then
        echo "Stopping Colab session ${SESSION}..."
        "${COLAB_CMD[@]}" stop -s "${SESSION}" >/dev/null 2>&1 || true
    fi
    exit "${exit_code}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$(uname -s)" != "Linux" && "$(uname -s)" != "Darwin" ]]; then
    echo "The official Colab CLI supports Linux and macOS only."
    echo "On Windows, run this script inside WSL2."
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required."
    exit 1
fi

if command -v colab >/dev/null 2>&1; then
    COLAB_BIN="$(command -v colab)"
elif command -v uv >/dev/null 2>&1; then
    echo "Installing the official Google Colab CLI with uv..."
    uv tool install google-colab-cli
    UV_TOOL_BIN_DIR="$(uv tool dir --bin)"
    COLAB_BIN="${UV_TOOL_BIN_DIR}/colab"
else
    CLI_VENV="${XDG_DATA_HOME:-${HOME}/.local/share}/meritkv-colab-cli-venv"
    echo "Installing the official Google Colab CLI into ${CLI_VENV}..."
    python3 -m venv "${CLI_VENV}"
    "${CLI_VENV}/bin/python" -m pip install --upgrade pip google-colab-cli
    COLAB_BIN="${CLI_VENV}/bin/colab"
fi

if [[ ! -x "${COLAB_BIN}" ]]; then
    echo "Could not locate the Colab CLI executable after installation."
    exit 1
fi

# google-colab-cli 0.6.0 can resolve the incompatible PyPI package
# jupyter-kernel-client 1.0.1 even though the official source locks Google's
# fork. Verify the API before allocating paid/limited accelerator time and
# repair only the isolated CLI environment when necessary.
CLI_PYTHON="$(head -n 1 "${COLAB_BIN}")"
CLI_PYTHON="${CLI_PYTHON#\#!}"
if [[ ! -x "${CLI_PYTHON}" ]]; then
    echo "Could not identify the Python interpreter used by ${COLAB_BIN}."
    exit 1
fi
if ! "${CLI_PYTHON}" -c \
    "import jupyter_kernel_client as j; assert hasattr(j, 'KernelClient')" \
    >/dev/null 2>&1; then
    echo "Repairing the Colab CLI's jupyter-kernel-client dependency from the official pinned Google fork..."
    PINNED_KERNEL_CLIENT="jupyter-kernel-client @ git+https://github.com/googlecolab/jupyter-kernel-client.git@f18e982c3265df5e923aa9def101ab3fd737e139"
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "${CLI_PYTHON}" --reinstall \
            "${PINNED_KERNEL_CLIENT}"
    else
        "${CLI_PYTHON}" -m pip install --force-reinstall \
            "${PINNED_KERNEL_CLIENT}"
    fi
fi

COLAB_CMD=("${COLAB_BIN}" "--auth=${COLAB_AUTH}")
COLAB_VERSION_OUTPUT="$("${COLAB_CMD[@]}" version)"
echo "${COLAB_VERSION_OUTPUT}"
CLI_VERSION_FILE="$(mktemp)"
printf '%s\n' "${COLAB_VERSION_OUTPUT}" > "${CLI_VERSION_FILE}"
MODEL_SELECTION_FILE="$(mktemp)"
printf '%s\n' "${MODEL_SELECTION}" > "${MODEL_SELECTION_FILE}"
ATTENTION_SELECTION_FILE="$(mktemp)"
printf '%s\n' "${ATTENTION_SELECTION}" > "${ATTENTION_SELECTION_FILE}"

if [[ -n "${HF_TOKEN:-}" ]]; then
    TOKEN_VALUE="${HF_TOKEN}"
    unset HF_TOKEN
else
    read -r -s -p "Hugging Face read token (required for gated Gemma): " TOKEN_VALUE
    echo
fi
if [[ -z "${TOKEN_VALUE}" ]]; then
    echo "A Hugging Face token is required."
    exit 1
fi

umask 077
TOKEN_FILE="$(mktemp)"
printf '%s' "${TOKEN_VALUE}" > "${TOKEN_FILE}"
unset TOKEN_VALUE

echo "Provisioning Colab T4 session ${SESSION}..."
SESSION_STARTED=1
"${COLAB_CMD[@]}" new -s "${SESSION}" --gpu T4
"${COLAB_CMD[@]}" status -s "${SESSION}"

printf '%s\n' \
    "from pathlib import Path; Path('${REMOTE_DIR}').mkdir(parents=True, exist_ok=True)" \
    | "${COLAB_CMD[@]}" exec -s "${SESSION}"

echo "Uploading validator files..."
"${COLAB_CMD[@]}" upload -s "${SESSION}" \
    "${PACKAGE_DIR}/validate_exact_splice.py" \
    "${REMOTE_DIR}/validate_exact_splice.py"
"${COLAB_CMD[@]}" upload -s "${SESSION}" \
    "${PACKAGE_DIR}/summarize_validation.py" \
    "${REMOTE_DIR}/summarize_validation.py"
"${COLAB_CMD[@]}" upload -s "${SESSION}" \
    "${PACKAGE_DIR}/run_remote_validation.py" \
    "${REMOTE_DIR}/run_remote_validation.py"
if [[ "${REMOTE_DRIVER}" != "run_remote_validation.py" ]]; then
    "${COLAB_CMD[@]}" upload -s "${SESSION}" \
        "${PACKAGE_DIR}/${REMOTE_DRIVER}" \
        "${REMOTE_DIR}/${REMOTE_DRIVER}"
fi
"${COLAB_CMD[@]}" upload -s "${SESSION}" \
    "${PACKAGE_DIR}/requirements-colab.txt" \
    "${REMOTE_DIR}/requirements-colab.txt"
"${COLAB_CMD[@]}" upload -s "${SESSION}" \
    "${CLI_VERSION_FILE}" \
    "${REMOTE_DIR}/colab_cli_version.txt"
rm -f -- "${CLI_VERSION_FILE}"
CLI_VERSION_FILE=""
"${COLAB_CMD[@]}" upload -s "${SESSION}" \
    "${MODEL_SELECTION_FILE}" \
    "${REMOTE_DIR}/model_selection.txt"
rm -f -- "${MODEL_SELECTION_FILE}"
MODEL_SELECTION_FILE=""
"${COLAB_CMD[@]}" upload -s "${SESSION}" \
    "${ATTENTION_SELECTION_FILE}" \
    "${REMOTE_DIR}/attention_selection.txt"
rm -f -- "${ATTENTION_SELECTION_FILE}"
ATTENTION_SELECTION_FILE=""
"${COLAB_CMD[@]}" upload -s "${SESSION}" \
    "${TOKEN_FILE}" \
    "${REMOTE_DIR}/.hf_token"
rm -f -- "${TOKEN_FILE}"
TOKEN_FILE=""

echo "Installing the recorded Transformers stack (Colab CUDA PyTorch is retained)..."
"${COLAB_CMD[@]}" install -s "${SESSION}" \
    -r "${PACKAGE_DIR}/requirements-colab.txt"

echo "Running validation for models ${MODEL_SELECTION} with attention ${ATTENTION_SELECTION}"
set +e
"${COLAB_CMD[@]}" exec -s "${SESSION}" \
    -f "${PACKAGE_DIR}/${REMOTE_DRIVER}"
REMOTE_STATUS=$?
set -e

mkdir -p -- "${LOCAL_RESULTS_DIR}"
echo "Downloading combined result archive..."
"${COLAB_CMD[@]}" download -s "${SESSION}" \
    "${REMOTE_ARCHIVE}" "${LOCAL_ARCHIVE}"

if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${LOCAL_ARCHIVE}"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${LOCAL_ARCHIVE}"
fi

echo "Downloaded: ${LOCAL_ARCHIVE}"
if [[ "${REMOTE_STATUS}" != "0" ]]; then
    echo "The remote driver reported an execution problem. The downloaded ZIP contains RUN_STATUS.json and retained logs."
fi
exit "${REMOTE_STATUS}"
