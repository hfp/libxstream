#!/usr/bin/env sh
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
FLOCK=$(command -v flock)

if [ -d "$1" ]; then
  ABSDIR=$(cd "$1" && pwd -P)
elif [ -f "$1" ]; then
  ABSDIR=$(cd "$(dirname "$1")" && pwd -P)
else
  ABSDIR=$(cd "$(dirname "$0")" && pwd -P)
fi

shift
cd "${ABSDIR}" || true
if [ "${FLOCK}" ]; then
  ${FLOCK} "${ABSDIR}" -c "$@"
else
  eval "$*"
fi

