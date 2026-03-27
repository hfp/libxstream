#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
MKTEMP=$(command -v mktemp)
MV=$(command -v mv)

if [ "${MKTEMP}" ] && [ "${MV}" ]; then
  TEMPLATE=${1/XXXXXX/}.XXXXXX
  TMPFILE=$(${MKTEMP} "${TEMPLATE}")
  EXTFILE=${TMPFILE: -6}
  NEWFILE=${1/XXXXXX/${EXTFILE}}
  if [ "$1" != "${NEWFILE}" ]; then
    ${MV} "${TMPFILE}" "${NEWFILE}"
    echo "${NEWFILE}"
  else
    echo "${TMPFILE}"
  fi
else
  touch "$1"
fi

