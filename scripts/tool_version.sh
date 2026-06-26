#!/usr/bin/env sh
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
HEAD=$(command -v head)
CUT=$(command -v cut)
GIT=$(command -v git)
TR=$(command -v tr)
GREP=$(command -v grep)

PRFIX=$1
CMPNT=${2:-0}
SHIFT=${3:-0}

if [ ! "${HEAD}" ] || [ ! "${CUT}" ] || [ ! "${TR}" ] || [ ! "${GREP}" ]; then
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 1
VERSION_FILE="${ROOT}/VERSION"
if [ ! -r "${VERSION_FILE}" ]; then
  >&2 echo "ERROR: missing ${VERSION_FILE}!"
  exit 1
fi

VERSION_BASE=$(${HEAD} -n1 "${VERSION_FILE}" | ${TR} -d '\r')
if ! printf '%s\n' "${VERSION_BASE}" | ${GREP} -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
  >&2 echo "ERROR: invalid version '${VERSION_BASE}' in ${VERSION_FILE}; expected MAJOR.MINOR.PATCH"
  exit 1
fi

if [ "${PRFIX}" ]; then
  PREFIX="$(echo "${PRFIX}_" | ${TR} a-z A-Z)"
fi

BRANCH=
REVC=0
if [ "${GIT}" ] && ${GIT} -C "${ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  BRANCH=$(${GIT} -C "${ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null)
  ANCHOR=$(${GIT} -C "${ROOT}" describe --tags --abbrev=0 --match '[0-9]*.[0-9]*.[0-9]*' 2>/dev/null)
  if [ "${ANCHOR}" ]; then
    REVC=$(${GIT} -C "${ROOT}" rev-list --count --no-merges "${ANCHOR}"..HEAD 2>/dev/null)
  else
    REVC=$(${GIT} -C "${ROOT}" rev-list --count --no-merges HEAD 2>/dev/null)
  fi
fi

PATCH=$((${REVC:-0}+SHIFT))
if [ "0" != "${PATCH}" ]; then
  EXT="-${PATCH}"
fi

if [ "0" = "${CMPNT}" ]; then
  if [ "${BRANCH}" ]; then
    echo "${BRANCH}-${VERSION_BASE}${EXT}"
  else
    echo "${VERSION_BASE}${EXT}"
  fi
elif [ "0" != "$((0>CMPNT))" ]; then
  MAJOR=$(echo "${VERSION_BASE}" | ${CUT} -d. -f1)
  MINOR=$(echo "${VERSION_BASE}" | ${CUT} -d. -f2)
  UPDTE=$(echo "${VERSION_BASE}" | ${CUT} -d. -f3)
  VERSION="${VERSION_BASE}${EXT}"
  echo "#ifndef ${PREFIX}VERSION_H"
  echo "#define ${PREFIX}VERSION_H"
  echo
  echo "#define ${PREFIX}BRANCH \"${BRANCH}\""
  echo "#define ${PREFIX}VERSION \"${VERSION}\""
  echo "#define ${PREFIX}VERSION_MAJOR  ${MAJOR:-0}"
  echo "#define ${PREFIX}VERSION_MINOR  ${MINOR:-0}"
  echo "#define ${PREFIX}VERSION_UPDATE ${UPDTE:-0}"
  echo "#define ${PREFIX}VERSION_PATCH  ${PATCH:-0}"
  echo "#define ${PREFIX}BUILD_DATE $(date +%Y%m%d)"
  echo
  echo "#endif"
elif [ "0" != "$((3>=CMPNT))" ]; then
  VALUE=$(echo "${VERSION_BASE}" | ${CUT} -d. -f"${CMPNT}")
  echo "${VALUE:-0}"
else
  echo "${PATCH}"
fi
