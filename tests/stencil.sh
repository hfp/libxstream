#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Runs the stencil sample with the kernel translated for the host (OCL=0), so
# no GPU and no OpenCL runtime are needed. Every case validates against the
# sample's own reference and the margin is checked here, because the sample
# reports the deviation rather than judging it.
#
# The point is coverage of the kernel and of the launcher, hence small grids and
# few steps: it is meant to run under a sanitizer, where SANITIZE (undefined,
# address, or both) reaches this script through the environment and Makefile.inc.
# Only the sample is built here, so a prebuilt LIBXS has to carry the same
# setting; a mismatch shows up as undefined __asan_* symbols at link time.
#
# Ragged extents are deliberate: 40 leaves a tail along the slow axis and 37
# along all three, which is where the tile bounds get interesting.
###############################################################################

HERE=$(cd "$(dirname "$0")" && pwd -P)
SAMPLE=${HERE}/../samples/stencil
GREP=$(command -v grep)
AWK=$(command -v awk)
MKTEMP=$(command -v mktemp)
MAKE=${MAKE:-$(command -v make)}
WC=$(command -v wc)
TR=$(command -v tr)

if [ ! "${GREP}" ] || [ ! "${AWK}" ] || [ ! "${MKTEMP}" ] || [ ! "${MAKE}" ] \
|| [ ! "${WC}" ] || [ ! "${TR}" ]; then
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
if [ ! -e "${SAMPLE}/stencil_cpu.c" ]; then
  >&2 echo "ERROR: ${SAMPLE} is not the stencil sample!"
  exit 1
fi

# Bound the threads: unbound ones undo the launcher's first-touch placement.
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
export OMP_PLACES=${OMP_PLACES:-cores}
# A sanitizer finding prints and continues by default, which would pass here.
export UBSAN_OPTIONS=${UBSAN_OPTIONS:-halt_on_error=1:print_stacktrace=1}
export STENCIL_CHECK=1

# Fields: label | CPUDEF | environment | arguments | margin
# Cases sharing a CPUDEF are adjacent, so each build serves all of them. A loose
# margin means the case is about running clean rather than about accuracy: a
# packed storage format, a compact operator and an absorbing boundary all
# deviate from the reference by construction.
CASES=(
  "fp32-32|-DSTENCIL_LAYOUT=0||-n 32 -t 2 -d 3|1e-5"
  "fp32-40|-DSTENCIL_LAYOUT=0||-n 40 -t 2 -d 3|1e-5"
  "fp32-37|-DSTENCIL_LAYOUT=0||-n 37 -t 2 -d 3|1e-5"
  "bf16s-1|-DSTENCIL_LAYOUT=0|STENCIL_BF16S=1|-n 32 -t 2 -d 3|1.0"
  "bf16s-2|-DSTENCIL_LAYOUT=0|STENCIL_BF16S=2|-n 32 -t 2 -d 3|1e-1"
  "fp16s|-DSTENCIL_LAYOUT=0|STENCIL_FP16S=1|-n 32 -t 2 -d 3|1.0"
  # ZYX needs a halo of at least one for the sample's staging to line up with
  # the reference, and a halo covering the gather selects the padded kernel.
  "zyx-halo1|-DSTENCIL_LAYOUT=2 -DSTENCIL_CPU_COMPACT=1|STENCIL_HALO=1|-n 32 -t 2 -d 3|1e-5"
  "zyx-halo4|-DSTENCIL_LAYOUT=2 -DSTENCIL_CPU_COMPACT=1|STENCIL_HALO=4|-n 32 -t 2 -d 3|1.0"
  "zyx-bf16s|-DSTENCIL_LAYOUT=2 -DSTENCIL_CPU_COMPACT=1|STENCIL_HALO=1 STENCIL_BF16S=2|-n 32 -t 2 -d 3|1.0"
  "compact-r1|-DSTENCIL_LAYOUT=2 -DSTENCIL_CPU_COMPACT=1|STENCIL_HALO=1 STENCIL_METHOD=1|-n 32 -t 2 -d 3|1.0"
  "compact-r2|-DSTENCIL_LAYOUT=2 -DSTENCIL_CPU_COMPACT=1|STENCIL_HALO=1 STENCIL_METHOD=2|-n 32 -t 2 -d 3|1.0"
  "compact-fit|-DSTENCIL_LAYOUT=2 -DSTENCIL_CPU_COMPACT=1|STENCIL_HALO=1 STENCIL_METHOD=3|-n 32 -t 2 -d 3|1.0"
  "pml|-DSTENCIL_LAYOUT=2 -DSTENCIL_PML=1|STENCIL_HALO=1|-n 32 -t 2 -d 3|1.0"
  "pml-fp16s|-DSTENCIL_LAYOUT=2 -DSTENCIL_PML=1|STENCIL_HALO=1 STENCIL_FP16S=1|-n 32 -t 2 -d 3|1.0"
  # The other two work-group models: a 1x1 group, and lanes as an OpenMP team
  # with real barriers. The team runs one work-group at a time, hence one step.
  "wg1x1|-DSTENCIL_LAYOUT=0 -DSTENCIL_CPU_LANES=0||-n 32 -t 2 -d 3|1e-5"
  "wgteam|-DSTENCIL_LAYOUT=0 -DSTENCIL_CPU_LANES=0 -DLIBXSTREAM_CPU_TEAM=1||-n 32 -t 1 -d 3|1e-5"
)

if [ "$*" ]; then
  SELECT="$*"
else # every label, space-separated to match the membership test below
  SELECT=""
  for CASE in "${CASES[@]}"; do SELECT="${SELECT} ${CASE%%|*}"; done
fi

BLDDIR=$(${MKTEMP} -d)
trap 'rm -rf "${BLDDIR}"' EXIT

echo "======================="
echo "Running stencil (OCL=0)"
echo "======================="

NTEST=1
NMAX=$(echo ${SELECT} | ${WC} -w | ${TR} -d " ")
BUILT=""
for CASE in "${CASES[@]}"; do
  IFS='|' read -r LABEL CPUDEF ENVIRON ARGS MARGIN <<<"${CASE}"
  if ! echo " ${SELECT} " | ${GREP} -q " ${LABEL} "; then continue; fi
  printf "%02d of %02d: %-12s " "${NTEST}" "${NMAX}" "${LABEL}"
  if [ "${BUILT}" != "${CPUDEF}" ]; then
    ERROR=$(${MAKE} -C "${SAMPLE}" -j "$(nproc 2>/dev/null || echo 2)" \
      OCL=0 BLDDIR="${BLDDIR}/obj" OUTDIR="${BLDDIR}" \
      CPUDEF="${CPUDEF}" 2>&1) || {
        echo "FAILED (build)"; >&2 echo "${ERROR}"; exit 1; }
    BUILT=${CPUDEF}
  fi
  RESULT=0
  # shellcheck disable=SC2086
  OUTPUT=$(env ${ENVIRON} "${BLDDIR}/stencil.x" ${ARGS} 2>&1) || RESULT=$?
  if [ 0 != ${RESULT} ]; then
    echo "FAILED (${RESULT})"; >&2 echo "${OUTPUT}"; exit ${RESULT}
  fi
  # shellcheck disable=SC2016
  LINF=$(${GREP} "Linf rel" <<<"${OUTPUT}" | ${AWK} '{print $3}')
  if ! ${AWK} -v v="${LINF}" -v m="${MARGIN}" \
    'BEGIN{exit !(v+0==v && ""!=v && v<=m+0)}'
  then
    echo "FAILED (Linf rel ${LINF:-missing} > ${MARGIN})"
    >&2 echo "${OUTPUT}"
    exit 1
  fi
  echo "OK (Linf rel ${LINF})"
  NTEST=$((NTEST+1))
done

# A label that matches nothing must not pass by running nothing.
if [ $((NTEST-1)) != "${NMAX}" ]; then
  >&2 echo "ERROR: ${SELECT} does not name $((NMAX-NTEST+1)) of the cases!"
  exit 1
fi
