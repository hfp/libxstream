#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################

HERE=$(cd "$(dirname "$0")" && pwd -P)
BLDDIR=${BLDDIR:-${HERE}}
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

# Tests left out unless named: stencil runs long and has its own CI step.
TESTS_DISABLED=${TESTS_DISABLED-stencil}

# A translation unit with a main function is a test, and so is every script but
# this one; a script is picked up by existing rather than by being listed.
if [ ! "$*" ]; then
  TESTS=$(cd "${HERE}" && ${GREP} -l "main[[:space:]]*(.*)" ./*.c 2>/dev/null)
  for SCRIPT in "${HERE}"/*.sh; do
    NAME=$(${SED} <<<"${SCRIPT}" 's/.*\///;s/\(.*\)\..*/\1/')
    if [ "test" != "${NAME}" ] && [ -e "${SCRIPT}" ] && \
       ! ${GREP} -qw "${NAME}" <<<"${TESTS_DISABLED}";
    then
      TESTS="${TESTS} ./${NAME}.sh"
    fi
  done
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
  printf "%02d of %02d: %-16s " "${NTEST}" "${NMAX}" "${NAME}"
  if [[ "${TEST}" == *.sh ]]; then
    TESTX="${HERE}/${NAME}.sh"
  elif [ -e "${BLDDIR}/${NAME}${EXE}" ]; then
    TESTX="${BLDDIR}/${NAME}${EXE}"
  elif [ -e "${HERE}/${NAME}.sh" ]; then
    TESTX="${HERE}/${NAME}.sh"
  else
    TESTX="${HERE}/${NAME}${EXE}"
  fi
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
