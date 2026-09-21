#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Checks which built-in device entry a device name selects, which decides the
# tuned parameters and the prediction model the SMM sample uses. The selection
# is the one the library makes (libxstream_opencl_device_match), shown by the
# sample's acc_match.x; no GPU and no OpenCL device are needed.
#
# Two kinds of cases. Every built-in entry must select itself, given as is and
# without its "[0x...]" (NVIDIA and AMD then select by the name's UID, Intel by
# words); this is derived from the entries and needs no upkeep. And names that
# have no entry of their own must land on the right relative, which is where a
# word matcher goes wrong: one that sent an H200 to the A100's parameters still
# passed every self-selection.
###############################################################################

HERE=$(cd "$(dirname "$0")" && pwd -P)
SAMPLE=${HERE}/../samples/smm
GREP=$(command -v grep)
SED=$(command -v sed)
MKTEMP=$(command -v mktemp)
MAKE=${MAKE:-$(command -v make)}

if [ ! "${GREP}" ] || [ ! "${SED}" ] || [ ! "${MKTEMP}" ] || [ ! "${MAKE}" ]; then
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
if [ ! -e "${SAMPLE}/acc_match.c" ]; then
  >&2 echo "ERROR: ${SAMPLE} is not the SMM sample!"
  exit 1
fi

# Fields: device name | part of the entry it must select ("none": no entry)
CASES=(
  "NVIDIA H100 NVL|NVIDIA H100"
  "NVIDIA H200|GH200"
  "NVIDIA A100-SXM4-40GB|A100"
  "Tesla V100-SXM2-32GB|V100"
  "Intel(R) Data Center GPU Max 1100|Max 1550"
  "Intel(R) Graphics [0x0bd5]|Max 1550"
  "Intel(R) Arc(TM) B570 Graphics|B580"
  "gfx90a:sramecc+:xnack-|gfx90a"
  "Some Unknown Device|none"
)

BLDDIR=$(${MKTEMP} -d)
trap 'rm -rf "${BLDDIR}"' EXIT

echo "========================"
echo "Running device selection"
echo "========================"

# acc_match.x uses neither BLAS nor OpenMP, which the sample otherwise wants:
# without them, this test needs nothing the rest of the suite does not.
ERROR=$(${MAKE} -C "${SAMPLE}" -j "$(nproc 2>/dev/null || echo 2)" \
  BLAS=0 OMP=0 BLDDIR="${BLDDIR}/obj" OUTDIR="${BLDDIR}" match 2>&1) || {
    >&2 echo "ERROR: failed to build acc_match.x"; >&2 echo "${ERROR}"; exit 1; }
MATCH=${BLDDIR}/acc_match.x

# The selection of a device name, e.g., "[6] NVIDIA H100 PCIe [0xa32d]".
select_entry() {
  "${MATCH}" "$1" | ${SED} -n "s/^selected: //p"
}

# The built-in entries, one per line, as the table lists them.
ENTRIES=$("${MATCH}" "-" | ${SED} -n "s/^\[[0-9]*\] [^ ]* [^ ]* [^ ]* //p")
if [ ! "${ENTRIES}" ]; then
  >&2 echo "ERROR: no built-in device entries!"
  exit 1
fi

NFAIL=0
NPASS=0
NSKIP=0
check() { # label, got, expected (exact), or expected part with 4th argument
  if { [ ! "$4" ] && [ "$2" = "$3" ]; } || { [ "$4" ] && [[ "$2" == *"$3"* ]]; }; then
    NPASS=$((NPASS+1))
  else
    >&2 printf "FAILED: %-38s selected %s\n" "$1" "${2:-nothing}"
    NFAIL=$((NFAIL+1))
  fi
}

I=0
while read -r ENTRY; do
  check "${ENTRY}" "$(select_entry "${ENTRY}")" "[${I}] ${ENTRY}"
  BARE=$(${SED} "s/ *\[0x[0-9a-fA-F]*\]$//" <<<"${ENTRY}")
  # a bare name shared by two entries (one name, two IDs) cannot select both
  if [ "${BARE}" != "${ENTRY}" ] && \
     [ 1 = "$(${SED} "s/ *\[0x[0-9a-fA-F]*\]$//" <<<"${ENTRIES}" | ${GREP} -cxF "${BARE}")" ];
  then
    check "${BARE}" "$(select_entry "${BARE}")" "[${I}] ${ENTRY}"
  fi
  I=$((I+1))
done <<<"${ENTRIES}"
echo "self-selection: ${NPASS} checks over ${I} entries (as is and bare)"

NSELF=${NPASS}
for CASE in "${CASES[@]}"; do
  IFS='|' read -r DEVICE EXPECT <<<"${CASE}"
  if [ "none" = "${EXPECT}" ]; then
    check "${DEVICE}" "$(select_entry "${DEVICE}")" "none"
  elif ${GREP} -qF "${EXPECT}" <<<"${ENTRIES}"; then
    check "${DEVICE}" "$(select_entry "${DEVICE}")" "${EXPECT}" part
  else # a build with fewer parameter sets has no such entry
    NSKIP=$((NSKIP+1))
  fi
done
echo "unseen names: $((NPASS-NSELF)) passed, ${NSKIP} skipped (no entry)"
if [ 0 != "${NSKIP}" ]; then # the suite shows stderr beside a pass
  >&2 echo "${NSKIP} unseen name(s) skipped: no entry built in"
fi

if [ 0 != "${NFAIL}" ]; then
  >&2 echo "ERROR: ${NFAIL} selection(s) failed!"
  exit 1
fi
