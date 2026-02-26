#!/usr/bin/env sh
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
SORT=$(command -v sort)
HEAD=$(command -v head)
CUT=$(command -v cut)
GIT=$(command -v git)

CMPNT=${1:-0}
SHIFT=${2:-0}

if [ "${SORT}" ] && [ "${HEAD}" ]; then
  TAG=$(${GIT} tag | ${SORT} -nr -t. -k1,1 -k2,2 -k3,3 | ${HEAD} -n1)
fi

BRANCH=$(${GIT} rev-parse --abbrev-ref HEAD 2>/dev/null)
BRANCH=${BRANCH:-unknown}

if [ "${TAG}" ]; then
  REVC=$(${GIT} rev-list --count --no-merges "${TAG}"..HEAD 2>/dev/null)
  BASE="-${TAG}"
else
  REVC=$(${GIT} rev-list --count --no-merges HEAD 2>/dev/null)
  BASE=""
fi

PATCH=$((${REVC:-0}+SHIFT))
if [ "0" != "${PATCH}" ]; then
  EXT="-${PATCH}"
fi

if [ "0" = "${CMPNT}" ]; then
  echo "${BRANCH}${BASE}${EXT}"
elif [ "0" != "$((0>CMPNT))" ]; then
  MAJOR=$(echo "${TAG}" | ${CUT} -d. -f1)
  MINOR=$(echo "${TAG}" | ${CUT} -d. -f2)
  UPDTE=$(echo "${TAG}" | ${CUT} -d. -f3)
  if [ "${TAG}" ]; then
    VERSION="${TAG}${EXT}"
  else
    VERSION="${PATCH}"
  fi
  echo "#ifndef LIBXS_VERSION_H"
  echo "#define LIBXS_VERSION_H"
  echo
  echo "#define LIBXS_BRANCH \"${BRANCH}\""
  echo "#define LIBXS_VERSION \"${VERSION}\""
  echo "#define LIBXS_VERSION_MAJOR  ${MAJOR:-0}"
  echo "#define LIBXS_VERSION_MINOR  ${MINOR:-0}"
  echo "#define LIBXS_VERSION_UPDATE ${UPDTE:-0}"
  echo "#define LIBXS_VERSION_PATCH  ${PATCH:-0}"
  echo "#define LIBXS_BUILD_DATE $(date +%Y%m%d)"
  echo
  echo "#endif"
elif [ "0" != "$((3>=CMPNT))" ]; then
  VALUE=$(echo "${TAG}" | ${CUT} -d. -f${CMPNT})
  echo "${VALUE:-0}"
else
  echo "${PATCH}"
fi
