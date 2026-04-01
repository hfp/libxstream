#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
BASENAME=$(command -v basename)
SED=$(command -v gsed)
CUT=$(command -v cut)

# GNU sed is desired (macOS)
if [ ! "${SED}" ]; then
  SED=$(command -v sed)
fi

if [ ! "${BASENAME}" ] || [ ! "${SED}" ] || [ ! "${CUT}" ]; then
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi

# find clang-format
CLANGFORMAT=$(command -v clang-format)
if [ ! "${CLANGFORMAT}" ]; then
  >&2 echo "ERROR: clang-format not found!"
  exit 1
fi

# determine version of clang-format
VERSION=$(${CLANGFORMAT} --version 2>/dev/null | \
  ${SED} -n "s/.* version \([0-9][0-9]*\)\..*/\1/p")
if [ ! "${VERSION}" ]; then
  >&2 echo "ERROR: cannot determine clang-format version!"
  exit 1
fi

# check for clang-format-V with V being VERSION+1..N
MAXVER=$((VERSION+10))
V=$((VERSION+1))
while [ "0" != "$((MAXVER>=V))" ]; do
  NEWER=$(command -v "clang-format-${V}")
  if [ "${NEWER}" ]; then
    CLANGFORMAT=${NEWER}
    VERSION=${V}
  fi
  V=$((V+1))
done

# forward all arguments to clang-format
exec ${CLANGFORMAT} "$@"
