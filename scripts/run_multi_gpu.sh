#!/usr/bin/env bash
# ===========================================================================
# run_multi_gpu.sh — Multi-GPU parallel operator test launcher
#
# Launches N copies of `python main.py`, each pinned to a separate GPU via
# platform-specific visible-devices env var.  After all processes finish,
# calls the aggregation script to summarize per-GPU results.
#
# Options:
#   --device <name>     Accelerator platform: nvidia|cambricon|ascend|metax|
#                       moore|iluvatar|kunlun|hygon|qy|ali (default: auto-detect)
#   --gpu-ids 2,3,4,5   Comma-separated GPU IDs to use
#   --gpu-count N        Use first N GPUs (default: all available)
#   --output-dir DIR     Base output directory (default: ./output)
#
# Examples:
#   # Auto-detect platform, use all GPUs
#   bash scripts/run_multi_gpu.sh test.json
#
#   # NVIDIA, specific GPUs
#   bash scripts/run_multi_gpu.sh test.json --device nvidia --gpu-ids 2,3,4,5
#
#   # Cambricon, all GPUs
#   bash scripts/run_multi_gpu.sh test.json --device cambricon
#
#   # Ascend, first 4 cards
#   bash scripts/run_multi_gpu.sh test.json --device ascend --gpu-count 4
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLATFORM_CONFIG="${SCRIPT_DIR}/platform_config.py"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_success() { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

usage() {
    cat <<EOF
Usage: bash $0 <test_config.json> [options]

Options:
  --device NAME      Accelerator: nvidia|cambricon|ascend|metax|moore|
                     iluvatar|kunlun|hygon|qy|ali  (default: auto-detect)
  --gpu-ids IDS      Comma-separated GPU IDs, e.g. --gpu-ids 2,3,4,5
  --gpu-count N      Use first N GPUs (default: all)
  --output-dir DIR   Base output directory (default: ./output)

Note: --gpu-ids takes priority over --gpu-count.

Examples:
  bash $0 test.json
  bash $0 test.json --device nvidia --gpu-ids 2,3,4,5
  bash $0 test.json --device cambricon --gpu-count 4
  bash $0 test.json --device ascend
EOF
    exit 1
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

TEST_CONFIG=""
DEVICE=""                # empty = auto-detect
GPU_IDS=""               # comma-separated
GPU_COUNT=0              # 0 = auto-detect
OUTPUT_DIR="./output"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        -*)
            log_error "Unknown option: $1"
            usage
            ;;
        *)
            if [[ -z "$TEST_CONFIG" ]]; then
                TEST_CONFIG="$1"
                shift
            else
                log_error "Unexpected argument: $1"
                usage
            fi
            ;;
    esac
done

if [[ -z "$TEST_CONFIG" ]]; then
    log_error "No test config JSON specified"
    usage
fi

if [[ ! -f "$TEST_CONFIG" ]]; then
    log_error "Test config not found: $TEST_CONFIG"
    exit 1
fi

# Make path absolute so subprocesses can find it
TEST_CONFIG="$(cd "$(dirname "$TEST_CONFIG")" && pwd)/$(basename "$TEST_CONFIG")"

# ---------------------------------------------------------------------------
# Resolve device platform
# ---------------------------------------------------------------------------

if [[ -z "$DEVICE" ]]; then
    DEVICE=$(python "${PLATFORM_CONFIG}" detect 2>/dev/null) || true
    if [[ -z "$DEVICE" || "$DEVICE" == "none" ]]; then
        log_error "Cannot auto-detect accelerator. Use --device <name> to specify."
        exit 1
    fi
    log_info "Auto-detected platform: ${DEVICE}"
fi

# Get platform-specific env var name
ENV_VAR=$(python "${PLATFORM_CONFIG}" env-var "${DEVICE}") || {
    log_error "Unknown device: ${DEVICE}"
    log_info "Supported devices:"
    python "${PLATFORM_CONFIG}" list
    exit 1
}

# ---------------------------------------------------------------------------
# Resolve GPU list
# ---------------------------------------------------------------------------

if [[ -n "$GPU_IDS" ]]; then
    # User specified exact IDs
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
elif [[ "$GPU_COUNT" -gt 0 ]]; then
    # User specified count, use first N
    GPU_ARRAY=($(seq 0 $((GPU_COUNT - 1))))
else
    # Auto-detect card count from platform SMI
    DETECTED_COUNT=$(python "${PLATFORM_CONFIG}" card-count "${DEVICE}")
    if [[ "${DETECTED_COUNT}" -lt 1 ]]; then
        log_error "No ${DEVICE} cards detected. Use --gpu-ids or --gpu-count."
        exit 1
    fi
    GPU_ARRAY=($(seq 0 $((DETECTED_COUNT - 1))))
fi

if [[ ${#GPU_ARRAY[@]} -lt 1 ]]; then
    log_error "No GPUs specified or detected"
    exit 1
fi

GPU_COUNT=${#GPU_ARRAY[@]}

# ---------------------------------------------------------------------------
# Prepare output directory
# ---------------------------------------------------------------------------

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MULTI_GPU_DIR="${OUTPUT_DIR}/multi_gpu_${TIMESTAMP}"
AGGREGATOR="${SCRIPT_DIR}/aggregate_multi_gpu.py"

log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Multi-GPU Operator Test"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Platform:  ${DEVICE}"
log_info "Env var:   ${ENV_VAR}"
log_info "Config:    ${TEST_CONFIG}"
log_info "GPUs:      ${GPU_IDS:-${GPU_COUNT} (auto)}"
log_info "Output:    ${MULTI_GPU_DIR}"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ---------------------------------------------------------------------------
# Launch parallel processes
# ---------------------------------------------------------------------------

declare -a PIDS=()
declare -a GPU_DIRS=()

log_info "Launching ${GPU_COUNT} processes..."

for gpu_id in "${GPU_ARRAY[@]}"; do
    GPU_DIR="${MULTI_GPU_DIR}/gpu_${gpu_id}"
    mkdir -p "${GPU_DIR}"
    GPU_DIRS+=("${GPU_DIR}")

    env "${ENV_VAR}=${gpu_id}" \
        python main.py "${TEST_CONFIG}" --output "${GPU_DIR}" \
        > "${GPU_DIR}/run.log" 2>&1 &

    PIDS+=($!)
    log_info "  GPU ${gpu_id}: PID $! → ${GPU_DIR}  (${ENV_VAR}=${gpu_id})"
done

echo ""
log_info "Waiting for all processes to complete..."

# ---------------------------------------------------------------------------
# Wait and collect results
# ---------------------------------------------------------------------------

SUCCESS_COUNT=0
FAIL_COUNT=0

for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    GPU_ID="${GPU_ARRAY[$i]}"
    if wait "${PID}"; then
        log_success "  GPU ${GPU_ID}: PASSED (PID ${PID})"
        ((SUCCESS_COUNT++)) || true
    else
        log_error "  GPU ${GPU_ID}: FAILED (PID ${PID})"
        tail -20 "${GPU_DIRS[$i]}/run.log" 2>/dev/null || true
        ((FAIL_COUNT++)) || true
    fi
done

echo ""
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Process Summary: ${SUCCESS_COUNT}/${GPU_COUNT} succeeded, ${FAIL_COUNT} failed"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------

if [[ "${FAIL_COUNT}" -gt 0 ]]; then
    log_warn "Some GPU tests failed — aggregation will include available results"
fi

if [[ -f "${AGGREGATOR}" ]]; then
    echo ""
    log_info "Running aggregation..."
    python "${AGGREGATOR}" "${MULTI_GPU_DIR}"
else
    log_warn "Aggregator not found at ${AGGREGATOR} — skipping aggregation"
    log_info "Individual results are in: ${MULTI_GPU_DIR}/gpu_*/"
fi

echo ""
log_info "All results saved to: ${MULTI_GPU_DIR}"

if [[ "${FAIL_COUNT}" -gt 0 ]]; then
    exit 1
fi
exit 0
