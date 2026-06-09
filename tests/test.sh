#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################

HERE=$(cd "$(dirname "$0")" && pwd -P)
GREP=$(command -v grep)
SED=$(command -v sed)
WC=$(command -v wc)
TR=$(command -v tr)

if [ ! "${GREP}" ] || [ ! "${SED}" ] || [ ! "${TR}" ] || [ ! "${WC}" ]; then
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi

if [ "Windows_NT" = "${OS}" ]; then
  export PATH=${PATH}:${HERE}/../lib
  EXE=.exe
else
  EXE=.x
fi

if [ ! "$*" ]; then
  TESTS=$(cd "${HERE}" && ${GREP} -l "main[[:space:]]*(.*)" ./*.c 2>/dev/null)
else
  TESTS="$*"
fi

echo "============="
echo "Running tests"
echo "============="

NTEST=1
NMAX=$(${WC} <<<"${TESTS}" -w | ${TR} -d " ")
for TEST in ${TESTS}; do
  NAME=$(${SED} <<<"${TEST}" 's/.*\///;s/\(.*\)\..*/\1/')
  printf "%02d of %02d: %-12s " "${NTEST}" "${NMAX}" "${NAME}"
  TESTX="${HERE}/${NAME}${EXE}"
  if [ -e "${TESTX}" ]; then
    RESULT=0
    ERROR=$({ \
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HERE}/../lib" \
      DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:${HERE}/../lib" \
      ${TESTX} >/dev/null; } 2>&1) || RESULT=$?
  else
    ERROR="Test is missing"
    RESULT=1
  fi
  if [ 0 != ${RESULT} ]; then
    echo "FAILED(${RESULT}) ${ERROR}"
    exit ${RESULT}
  else
    echo "OK ${ERROR}"
  fi
  NTEST=$((NTEST+1))
done
