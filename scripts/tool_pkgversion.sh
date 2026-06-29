#!/usr/bin/env sh
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Synchronize packaging metadata (RPM spec, Debian changelog) with the
# project version from VERSION. Only patches when the version actually
# changed. Run from the repository root:
#   sh scripts/tool_pkgversion.sh
###############################################################################
SED=$(command -v sed)
GREP=$(command -v grep)

if [ ! "${SED}" ] || [ ! "${GREP}" ]; then
  >&2 echo "ERROR: missing prerequisites (sed, grep)!"
  exit 1
fi

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "${HERE}/.." && pwd)

VERSION_FILE="${ROOT}/version.txt"
if [ ! -r "${VERSION_FILE}" ]; then
  >&2 echo "ERROR: missing ${VERSION_FILE}"
  exit 1
fi

VERSION=$(${SED} -n '1{s/\r$//;p;}' "${VERSION_FILE}")
if ! printf '%s\n' "${VERSION}" | ${GREP} -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
  >&2 echo "ERROR: invalid version '${VERSION}' in ${VERSION_FILE}"
  exit 1
fi

NAME=$(${GREP} -m1 'project(' "${ROOT}/CMakeLists.txt" \
  | ${SED} 's/.*project(//;s/[[:space:]].*//')

if [ -z "${NAME}" ]; then
  >&2 echo "ERROR: cannot extract project name from CMakeLists.txt"
  exit 1
fi

# --- RPM spec (Version: line) ---
SPEC="${HERE}/fedora/${NAME}.spec"
if [ -f "${SPEC}" ]; then
  OLD=$(${GREP} -m1 '^Version:' "${SPEC}" | ${SED} 's/[^0-9.]//g')
  if [ "${OLD}" != "${VERSION}" ]; then
    ${SED} -i "s/^Version:.*$/Version:        ${VERSION}/" "${SPEC}"
  fi
fi

# --- Debian changelog (version in first line) ---
DCHLOG="${HERE}/debian/changelog"
if [ -f "${DCHLOG}" ]; then
  OLD=$(${SED} -n '1s/.*(\(.*\)-[^-]*).*/\1/p' "${DCHLOG}")
  if [ "${OLD}" != "${VERSION}" ]; then
    ${SED} -i "1s/(.*)/(${VERSION}-1)/" "${DCHLOG}"
  fi
fi
