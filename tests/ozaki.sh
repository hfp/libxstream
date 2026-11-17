#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Runs the Ozaki sample (fp64) on every GPU in GPUS, which are simulated NVIDIA
# GPUs (VGPU_GPU, see "vgpu list-gpus") or "local" for whatever device is
# present, through every kernel level the GPU has: on NVIDIA LIBXSTREAM_NV
# (1 generic, 2 dp4a and m8n8k16, 3 m16n8k32, 4 m16n8k32 on Hopper and later)
# and warp-group MMA where it is taken (on a simulated GPU of sm_90), on Intel
# LIBXSTREAM_INTEL (0 generic, 1 Intel GPU, 2 DPAS), elsewhere the one generic
# level. A local device's vendor and top level are what the sample selects. Both
# schemes, with the carriers of Scheme 2 (int8, int8 with symmetric residues,
# and bf16 with symmetric residues, which needs warp-group MMA), at a size of
# whole tiles and at one that is not (SIZES).
#
# What ran is asserted rather than assumed, since a kernel that fails to build
# downgrades to another path whose result is still right. A run passes if it
# exits cleanly, prints no error, every kernel was built for the level it asked
# for (-DNV, -DINTEL), its configuration (OZAKI_VERBOSE=3) names the warp-group
# state it asked for, and its result is bit-identical to that of every other
# path of the same scheme and size on the same GPU (the DIFF line), which holds
# because the products are exact.
#
# On a simulated GPU, warp-group MMA runs from the PTX PoCL built
# (OZAKI_WGMMA=2): a first run leaves it in a kernel cache of the case's own,
# samples/ozaki/ozaki_splice.x patches it there, and the run from that cache is
# the one checked. The first run computes nothing by construction and has to be
# wrong, which shows that the patched kernel is what the second one ran.
#
# Every checked run on a simulated GPU is counted (scripts/tool_collect.sh
# count). Into LOGS (default: ./ozaki): the logs, counts.tsv (per kernel and
# launch), and summary.md (every run, and the instructions per kernel that
# changed against PREV, a counts.tsv of an earlier run). META names what the
# counts depend on (toolchain, simulator). The exit status is non-zero if a
# run failed.
###############################################################################

HERE=$(cd "$(dirname "$0")" && pwd -P)
ROOT=${HERE}/..
LOGS=${LOGS:-${PWD}/ozaki}
SIZES=${SIZES:-256 257}
MAKE=${MAKE:-$(command -v make)}
NPROC=$(nproc 2> /dev/null || echo 4)
GPUS=${GPUS:-nvidia/h100-pcie nvidia/gh200-480gb}
STATUS=0

mkdir -p "${LOGS}/logs" || exit 1
BIN=$(mktemp -d) || exit 1
trap 'rm -rf "${BIN}"' EXIT
export BIN # scripts/tool_collect.sh runs the binaries from there
COUNTS=${LOGS}/counts.tsv
RESULTS=${LOGS}/results.md
SUMMARY=${LOGS}/summary.md
{ echo "# ${META:-unknown}"; printf 'target\tkernel\tcategory\tvalue\n'; } > "${COUNTS}"
echo "| GPU | Case | Environment | Result |" > "${RESULTS}"
echo "|---|---|---|---|" >> "${RESULTS}"

"${MAKE}" -j "${NPROC}" -C "${ROOT}/samples/ozaki" || exit 1
"${MAKE}" -j "${NPROC}" -C "${ROOT}/samples/ozaki" splice || exit 1
cp "${ROOT}/samples/ozaki/ozaki.x" "${ROOT}/samples/ozaki/ozaki_splice.x" "${BIN}" || exit 1

# cases <var> <first> <top> <wgmma>: name|environment|size, levels first to top of
# the variable that selects them (none: one generic level), and warp-group MMA
# as OZAKI_WGMMA gives it (none: not taken); a name ends in the precision
cases() {
  local VAR=$1 TOP=$3 WG=$4 LEVEL N
  local TAG=${VAR#LIBXSTREAM_}
  TAG=${TAG,,}
  for N in ${SIZES}; do
    for LEVEL in $(seq "$2" "${TOP}"); do
      local SEL="${VAR}=${LEVEL}"
      if [ -z "${VAR}" ]; then SEL=""; TAG="generic"; LEVEL=""; fi
      if [ "LIBXSTREAM_NV" = "${VAR}" ] && [ 4 = "${LEVEL}" ]; then SEL="${SEL} OZAKI_WGMMA=0"; fi
      echo "oz1_${TAG}${LEVEL}_n${N}_double|OZAKI=1 ${SEL}|${N}"
      echo "oz2int8_${TAG}${LEVEL}_n${N}_double|OZAKI=2 ${SEL}|${N}"
      echo "oz2sym_${TAG}${LEVEL}_n${N}_double|OZAKI=2 OZAKI_SYMRES=1 ${SEL}|${N}"
      if [ -z "${VAR}" ]; then break; fi
    done
    if [ -n "${WG}" ]; then
      echo "oz2int8_wgmma_n${N}_double|OZAKI=2 ${VAR}=${TOP} OZAKI_WGMMA=${WG}|${N}"
      echo "oz2sym_wgmma_n${N}_double|OZAKI=2 OZAKI_SYMRES=1 ${VAR}=${TOP} OZAKI_WGMMA=${WG}|${N}"
      echo "oz2bf16_wgmma_n${N}_double|OZAKI=2 OZAKI_BF16=1 OZAKI_SYMRES=1 ${VAR}=${TOP} OZAKI_WGMMA=${WG}|${N}"
    fi
  done
}

# value <name> <environment>: the value a NAME=VALUE pair in the environment gives
value() {
  sed -n "s/.*\<$1=\([^ ]*\).*/\1/p" <<< "$2"
}

for GPU in ${GPUS}; do
  TARGET=${GPU//\//_}
  if [ "local" = "${GPU}" ]; then
    # the device's vendor and top level, as the sample configures itself by default
    unset VGPU_GPU POCL_CUDA_GPU_ARCH
    CONFIG=$(OZAKI=2 OZAKI_VERBOSE=3 "${BIN}/ozaki.x" 64 2>&1 | grep -m1 "^INFO OZAKI: gpu=")
    WG=""
    if [ -n "$(value intel "${CONFIG}")" ]; then
      VAR=LIBXSTREAM_INTEL FIRST=0 TOP=$(value intel "${CONFIG}")
    elif [ -n "$(value nv "${CONFIG}")" ]; then
      VAR=LIBXSTREAM_NV FIRST=1 TOP=$(value nv "${CONFIG}")
      if [ 1 = "$(value wgmma "${CONFIG}")" ]; then WG=1; fi
    else
      VAR="" FIRST=1 TOP=1
    fi
  else
    SM=$(vgpu info --gpu "${GPU}" | sed -n 's/.*compute capability:[[:space:]]*\([0-9]*\)\.\([0-9]*\).*/sm_\1\2/p')
    if [ -z "${SM}" ]; then
      echo "| ${GPU} | | | FAIL: not a simulated NVIDIA GPU |" >> "${RESULTS}"
      STATUS=1
      continue
    fi
    export VGPU_GPU=${GPU} POCL_CUDA_GPU_ARCH=${SM}
    VAR=LIBXSTREAM_NV FIRST=1 TOP=1 WG=""
    if [ 90 -le "${SM#sm_}" ]; then TOP=4; elif [ 80 -le "${SM#sm_}" ]; then TOP=3; elif [ 75 -le "${SM#sm_}" ]; then TOP=2; fi
    if [ sm_90 = "${SM}" ]; then WG=2; fi
  fi
  CACHE=${BIN}/cache-${TARGET}
  declare -A REFERENCE=()
  while IFS='|' read -r NAME SETENV N; do
    SETENV="${SETENV} OZAKI_VERBOSE=3"
    LOG=${LOGS}/logs/run-${TARGET}-${NAME}.log
    PRIME=${LOGS}/logs/prime-${TARGET}-${NAME}.log
    WGMMA=$(value OZAKI_WGMMA "${SETENV}")
    RESULT=""
    export POCL_CACHE_DIR=${CACHE}
    if [ 2 = "${WGMMA}" ]; then
      export POCL_CACHE_DIR=${CACHE}-${NAME}
      # shellcheck disable=SC2086
      env ${SETENV} "${BIN}/ozaki.x" "${N}" > "${PRIME}" 2>&1
      SPLICED=0
      while read -r PTX; do
        if "${BIN}/ozaki_splice.x" "${PTX}" "${PTX}" >> "${PRIME}" 2>&1; then SPLICED=$((SPLICED + 1)); fi
      done < <(grep -rl --include='*.ptx' "// WGMMA_SLOT n" "${POCL_CACHE_DIR}" 2> /dev/null)
      if [ 0 = "${SPLICED}" ]; then
        RESULT="FAIL: no warp-group kernel to patch"
      elif ! grep -q "^DIFF: .* -> " "${PRIME}"; then
        RESULT="FAIL: the unpatched kernel was not what ran"
      fi
    fi
    if [ -z "${RESULT}" ] && [ "local" = "${GPU}" ]; then
      CODE=0
      # shellcheck disable=SC2086
      env ${SETENV} "${BIN}/ozaki.x" "${N}" > "${LOG}" 2>&1 || CODE=$?
    elif [ -z "${RESULT}" ]; then
      printf '%s|ozaki.x|%s|%s\n' "${NAME}" "${SETENV}" "${N}" > "${BIN}/case"
      CASELIST=${BIN}/case "${ROOT}/scripts/tool_collect.sh" count "${BIN}/out" > /dev/null 2>&1
      mv "${BIN}/out/logs/count-${TARGET}-${NAME}.log" "${LOG}" 2> /dev/null
      if [ -s "${BIN}/out/counts.tsv" ]; then
        tail -n +2 "${BIN}/out/counts.tsv" >> "${COUNTS}"
      fi
      CODE=$(sed -n "s/^${NAME} //p" "${BIN}/out/status.txt" 2> /dev/null)
      rm -rf "${BIN}/out"
    fi
    if [ -z "${RESULT}" ]; then
      CONFIG=$(grep -m1 "^INFO OZAKI: gpu=" "${LOG}" 2> /dev/null)
      DIFF=$(grep -m1 "^DIFF: " "${LOG}" 2> /dev/null)
      LEVEL=$(value "${VAR:-LEVEL}" "${SETENV}")
      # the level every kernel was built for (-DNV, -DINTEL)
      BUILT=$(grep "^INFO OZAKI: -D" "${LOG}" 2> /dev/null | grep -o -- "-D${VAR#LIBXSTREAM_}=[0-9]*" | sort -u | paste -sd' ' -)
      KEY=$(value OZAKI "${SETENV}")_${N}
      if [ "0" != "${CODE}" ]; then
        RESULT="FAIL: exit status ${CODE:-unknown}"
      elif grep -q "ERROR" "${LOG}"; then
        RESULT="FAIL: $(grep -m1 "ERROR" "${LOG}")"
      elif [ -n "${VAR}" ] && [ "-D${VAR#LIBXSTREAM_}=${LEVEL}" != "${BUILT}" ]; then
        RESULT="FAIL: not built for level ${LEVEL} (${BUILT:-no build line})"
      elif [ "${WGMMA:-0}" != "$(value wgmma "${CONFIG}" | grep . || echo 0)" ]; then
        RESULT="FAIL: warp-group MMA is not ${WGMMA:-0} (${CONFIG#INFO OZAKI: })"
      elif [ -z "${DIFF}" ] || grep -q " -> " <<< "${DIFF}"; then
        RESULT="FAIL: wrong result (${DIFF:-none})"
      elif [ -z "${REFERENCE[${KEY}]}" ]; then
        REFERENCE[${KEY}]=${DIFF}
        RESULT="ok"
      elif [ "${REFERENCE[${KEY}]}" != "${DIFF}" ]; then
        RESULT="FAIL: differs from the first path (${DIFF#DIFF: })"
      else
        RESULT="ok"
      fi
    fi
    echo "| ${GPU} | ${NAME} | \`${SETENV% OZAKI_VERBOSE=3}\` | ${RESULT} |" >> "${RESULTS}"
    echo "${GPU} ${NAME}: ${RESULT}"
    if [ "ok" != "${RESULT}" ]; then STATUS=1; fi
  done < <(cases "${VAR}" "${FIRST}" "${TOP}" "${WG}")
  unset REFERENCE
done

{ echo "## Ozaki sample on ${GPUS}"
  echo
  echo "$(grep -c '| ok |$' "${RESULTS}") of $(($(wc -l < "${RESULTS}") - 2)) runs passed (${META:-unknown})."
  echo
  if [ -n "${PREV}" ] && [ -s "${PREV}" ]; then
    if [ "$(head -1 "${PREV}")" != "$(head -1 "${COUNTS}")" ]; then
      echo "**Counter changes**: the counts depend on what changed ($(head -1 "${PREV}" | sed 's/^# //') -> ${META:-unknown}), hence differences are expected and not listed."
    else
      CHANGES=$(awk -F'\t' '
        FNR <= 2 || "instructions" != $3 { next }
        NR == FNR { prev[$1 FS $2] = $4; next }
        ($1 FS $2) in prev && prev[$1 FS $2] != $4 {
          p = prev[$1 FS $2]
          printf "- %s %s: %s -> %s instructions (%+.1f%%)\n", $1, $2, p, $4, 0 < p ? 100 * ($4 - p) / p : 0
        }' "${PREV}" "${COUNTS}")
      if [ -n "${CHANGES}" ]; then echo "**Counter changes** since the previous passing run:"; echo "${CHANGES}"
      else echo "**Counter changes** since the previous passing run: none."; fi
    fi
    echo
  fi
  echo "<details><summary>Every run</summary>"
  echo
  cat "${RESULTS}"
  echo
  echo "</details>"
} > "${SUMMARY}"
exit ${STATUS}
