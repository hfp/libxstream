#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Runs the cases below (the named mixes and the samples) and collects, per
# kernel, either its time on a device or what a simulator counted for it, with
# the same arguments and the same kernel names, for scripts/tool_score.py:
#
#   tool_collect.sh time  OUT   times.csv: device, target, kernel, time, unit, source
#   tool_collect.sh count OUT   counts.tsv: target, kernel, category, value (per launch)
#
# "time" takes each kernel's time from LIBXSTREAM_PROFILE (device events), in
# milliseconds; DEVICE and TARGET name the device (default: the OpenCL device
# name, and "local"). "count" runs on the simulated GPU in VGPU_GPU, one
# work-group at a time (VGPU_THREADS=1), with the counts of every launch
# (VGPU_DISPATCH_STATS) and the name of every kernel (VGPU_TRACE_LAUNCHES).
#
# BIN holds the binaries: mixes_double.x, mixes_float.x, acc_bench_double.x,
# acc_bench_float.x, dbm_bench.x, ozaki.x, and stencil.x. CASES selects cases
# by name (default: all), NREPEAT the timed repetitions (default: 10, and 1
# for counting, which does not depend on it). A kernel is named by its case
# and its own name, "smm13x5x7_double:dsmm13x5x7_double" for instance, and a
# named mix by its mixture only ("stream_double"). A failing case is reported
# in OUT/status.txt and does not stop the others; the exit status says if any
# case failed.
###############################################################################

MODE=$1
OUT=$2
BIN=${BIN:-${PWD}}
STATUS=0

if [ "time" != "${MODE}" ] && [ "count" != "${MODE}" ] || [ -z "${OUT}" ]; then
  echo "Usage: $0 time|count OUT"
  exit 1
fi
if [ "count" = "${MODE}" ] && [ -z "${VGPU_GPU}" ]; then
  echo "ERROR: counting needs a simulated GPU (VGPU_GPU)"
  exit 1
fi
mkdir -p "${OUT}/logs" || exit 1

# name|binary|environment|arguments (NREPEAT is replaced); the name ends in the precision
ALL="
mixes_double|mixes_double.x||1048576 256 1024 NREPEAT
mixes_float|mixes_float.x||1048576 256 1024 NREPEAT
smm4x4x4_double|acc_bench_double.x||NREPEAT 10000 4x4x4
smm13x5x7_double|acc_bench_double.x||NREPEAT 10000 13x5x7
smm23x23x23_double|acc_bench_double.x||NREPEAT 10000 23x23x23
smm13x5x7_float|acc_bench_float.x||NREPEAT 10000 13x5x7
smm23x23x23_float|acc_bench_float.x||NREPEAT 10000 23x23x23
dbm13x5x7_double|dbm_bench.x|DBM_MULTIPLY_SMM=-1|NREPEAT 10000 13x5x7
dbm23x23x23_double|dbm_bench.x|DBM_MULTIPLY_SMM=-1|NREPEAT 10000 23x23x23
ozaki1n256_double|ozaki.x|OZAKI=1|256
ozaki2n256_double|ozaki.x|OZAKI=2|256
stencil64_float|stencil.x||-n 64 -t 4
"

if [ "time" = "${MODE}" ]; then
  NREPEAT=${NREPEAT:-10}
  DEVICE=${DEVICE:-$(clinfo --raw 2> /dev/null | sed -n 's/.*CL_DEVICE_NAME[[:space:]]*//p' | sed -n 1p)}
  TARGET=${TARGET:-local}
  RESULT_FILE=${OUT}/times.csv
  if [ ! -s "${RESULT_FILE}" ]; then echo "device,target,kernel,time,unit,source" > "${RESULT_FILE}"; fi
else
  NREPEAT=1
  TARGET=${VGPU_GPU}
  RESULT_FILE=${OUT}/counts.tsv
  if [ ! -s "${RESULT_FILE}" ]; then printf 'target\tkernel\tcategory\tvalue\n' > "${RESULT_FILE}"; fi
fi
echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) ${MODE} ${TARGET} ${DEVICE}" >> "${OUT}/status.txt"

# kernel <case> <raw name>: the name of a kernel in the results
kernel() {
  case $1 in
    mixes_*) echo "${2#mixes_}_${1##*_}" ;;
    *) echo "${1%_*}_${1##*_}:$2_${1##*_}" ;;
  esac
}

while IFS='|' read -r NAME EXE SETENV ARGS; do
  if [ -z "${NAME}" ]; then continue; fi
  if [ -n "${CASES}" ] && ! grep -qw "${NAME}" <<< "${CASES}"; then continue; fi
  LOG=${OUT}/logs/${MODE}-${TARGET//\//_}-${NAME}.log
  RESULT=0
  # shellcheck disable=SC2086
  if [ "time" = "${MODE}" ]; then
    env ${SETENV} LIBXSTREAM_PROFILE=1 "${BIN}/${EXE}" ${ARGS//NREPEAT/${NREPEAT}} > "${LOG}" 2>&1 || RESULT=$?
  else
    env ${SETENV} VGPU_THREADS=1 VGPU_DISPATCH_STATS=1 VGPU_TRACE_LAUNCHES=1 \
      "${BIN}/${EXE}" ${ARGS//NREPEAT/${NREPEAT}} > "${LOG}" 2>&1 || RESULT=$?
  fi
  echo "${NAME} ${RESULT}" >> "${OUT}/status.txt"
  if [ "0" != "${RESULT}" ]; then STATUS=1; continue; fi
  if [ "time" = "${MODE}" ]; then
    # "PROF ACC/OpenCL: ID=<id> <kernel>=<ms> ms"
    sed -n 's/^PROF ACC\/OpenCL: ID=[^ ]* \([^ =]*\)=\([0-9.]*\) ms.*/\1 \2/p' "${LOG}" |
      while read -r RAW MS; do
        printf '%s,%s,%s,%s,ms,measured\n' "${DEVICE}" "${TARGET}" "$(kernel "${NAME}" "${RAW}")" "${MS}"
      done >> "${RESULT_FILE}"
  else
    # every launch: its kernel's name (trace), then its counts; the counts per launch
    awk '
      /^VirtualGPU [A-Za-z]*: launch / { name[n++] = $4 }
      /^VGPU_DISPATCH / {
        k = name[m++]; ++launches[k]
        for (i = 2; i <= NF; ++i) { split($i, kv, "="); if ("device" != kv[1]) sum[k, kv[1]] += kv[2] }
      }
      END {
        for (key in sum) { split(key, kc, SUBSEP); printf "%s\t%s\t%.10g\n", kc[1], kc[2], sum[key] / launches[kc[1]] }
        for (k in launches) printf "%s\tlaunches\t%d\n", k, launches[k]
      }' "${LOG}" | sort |
      while IFS=$'\t' read -r RAW CATEGORY VALUE; do
        printf '%s\t%s\t%s\t%s\n' "${TARGET}" "$(kernel "${NAME}" "${RAW}")" "${CATEGORY}" "${VALUE}"
      done >> "${RESULT_FILE}"
  fi
done <<< "${ALL}"

exit ${STATUS}
