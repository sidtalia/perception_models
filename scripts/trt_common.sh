#!/usr/bin/env bash
# Shared TensorRT / DLA helpers for perception_models engine builds.
# Source from other scripts:  # shellcheck source=trt_common.sh
#   source "$(cd "$(dirname "$0")" && pwd)/trt_common.sh"
#
# Env:
#   USE_DLA=1|0     Prefer Jetson DLA (default 1). Layers that cannot run on DLA
#                   fall back to GPU when --allowGPUFallback is set.
#                   If TensorRT reports num_DLA_cores==0 (e.g. JetPack 7.2 /
#                   L4T R39.2 SBSA stack), DLA is skipped automatically.
#   DLA_CORE=0|1    Which DLA core (Orin has 0 and 1). Default 0.
#   USE_DLA_DYNAMIC=1|0
#                   Also pass DLA flags for engines with dynamic shapes
#                   (NanoSAM batch profiles). Default 1. Set 0 if trtexec
#                   rejects DLA+dynamic on your JetPack.
#
# Populates:
#   TRTEXEC         path to trtexec
#   TRT_DLA_FLAGS   bash array of extra flags (may be empty)
#   TRT_NUM_DLA_CORES  last probed count (empty until configure_trt_dla_flags)

resolve_trtexec() {
  TRTEXEC="$(command -v trtexec || true)"
  if [[ -z "${TRTEXEC}" && -x /usr/src/tensorrt/bin/trtexec ]]; then
    TRTEXEC=/usr/src/tensorrt/bin/trtexec
  fi
  if [[ -z "${TRTEXEC}" ]]; then
    echo "trtexec not found (install TensorRT / use mushr_jazzy)" >&2
    return 1
  fi
}

# Probe Builder.num_DLA_cores (TensorRT API). Returns count on stdout, or 0 on error.
probe_trt_num_dla_cores() {
  python3 - <<'PY' 2>/dev/null || echo 0
import tensorrt as trt
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
print(int(builder.num_DLA_cores))
PY
}

# Args: optional "dynamic" if this engine uses --min/opt/maxShapes.
configure_trt_dla_flags() {
  local kind="${1:-static}"
  TRT_DLA_FLAGS=()
  local use_dla="${USE_DLA:-1}"
  if [[ "${use_dla}" != "1" ]]; then
    echo "[trt] GPU only (USE_DLA=${use_dla})"
    return 0
  fi
  if [[ "${kind}" == "dynamic" && "${USE_DLA_DYNAMIC:-1}" != "1" ]]; then
    echo "[trt] GPU only for dynamic shapes (USE_DLA_DYNAMIC=0)"
    return 0
  fi

  TRT_NUM_DLA_CORES="$(probe_trt_num_dla_cores)"
  TRT_NUM_DLA_CORES="${TRT_NUM_DLA_CORES%%[^0-9]*}"
  TRT_NUM_DLA_CORES="${TRT_NUM_DLA_CORES:-0}"
  if [[ "${TRT_NUM_DLA_CORES}" -le 0 ]]; then
    echo "[trt] TensorRT reports 0 DLA cores — building GPU FP16"
    echo "[trt] (JetPack 7.2 / L4T R39.2: DLA not wired into TRT yet; cuDLA may still see HW)"
    return 0
  fi

  local core="${DLA_CORE:-0}"
  if ! [[ "${core}" =~ ^[0-9]+$ ]]; then
    echo "[trt] invalid DLA_CORE=${core}; using 0" >&2
    core=0
  fi
  if [[ "${core}" -ge "${TRT_NUM_DLA_CORES}" ]]; then
    echo "[trt] DLA_CORE=${core} >= num_DLA_cores=${TRT_NUM_DLA_CORES}; using 0" >&2
    core=0
  fi
  echo "[trt] DLA core ${core}/${TRT_NUM_DLA_CORES} + allowGPUFallback (${kind} engine)"
  TRT_DLA_FLAGS=(--useDLACore="${core}" --allowGPUFallback)
}
