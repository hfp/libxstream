#!/usr/bin/env sh
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# shellcheck disable=SC2086
MKDIR=$(command -v mkdir)
DIFF=$(command -v diff)
UNIQ=$(command -v uniq)
SED=$(command -v sed)
TR=$(command -v tr)

if [ "${MKDIR}" ] && [ "${SED}" ] && [ "${TR}" ] && [ "${DIFF}" ] && [ "${UNIQ}" ]; then
  # ensure proper permissions
  if [ "${UMASK}" ]; then
    UMASK_CMD="umask ${UMASK};"
    eval "${UMASK_CMD}"
  fi
  # STATENAME selects which state this invocation maintains, so a second scope
  # (e.g. link-only flags) can be tracked beside the default without the two
  # invalidating each other.
  if [ -z "${STATENAME}" ]; then STATENAME=.state; fi
  if [ "$1" ]; then
    STATEFILE=$1/${STATENAME}
    ${MKDIR} -p "$1"
    shift
  else
    STATEFILE=${STATENAME}
  fi
  # The trigger the make rule depends on. Touching this script instead would be
  # one trigger shared by every scope, so a link-only change would still
  # invalidate every object -- which is the whole point of having two.
  TRIGGER=${STATEFILE}.trigger

  STATE=$(${TR} '?' '\n' | ${TR} '"' \' | ${SED} -e 's/^ */\"/' -e 's/   */ /g' -e 's/ *$/\\n\"/')
  TOUCH=$(command -v touch)
  if [ -e "${STATEFILE}" ]; then
    if [ "$@" ]; then
      EXCLUDE="-e /\($(echo "$@" | ${SED} "s/[[:space:]][[:space:]]*/\\\|/g" | ${SED} "s/\\\|$//")\)/d"
    fi
    # BSD's diff does not support --unchanged-line-format=""
    STATE_DIFF=$(printf "%s\n" "${STATE}" \
               | ${DIFF} "${STATEFILE}" - 2>/dev/null | ${SED} -n 's/[<>] \(..*\)/\1/p' \
               | ${SED} -e 's/=..*$//' -e 's/\"//g' -e '/^$/d' ${EXCLUDE} | ${UNIQ})
    RESULT=$?
    if [ "0" != "${RESULT}" ] || [ "${STATE_DIFF}" ]; then
      if [ "" = "${NOSTATE}" ] || [ "0" = "${NOSTATE}" ]; then
        printf "%s\n" "${STATE}" >"${STATEFILE}"
      fi
      echo "${TRIGGER} ${STATE_DIFF}"
      # only needed to execute body of .state-rule
      if [ "${TOUCH}" ]; then ${TOUCH} "${TRIGGER}"; fi
    fi
  else # difference must not be determined
    if [ "" = "${NOSTATE}" ] || [ "0" = "${NOSTATE}" ]; then
      printf "%s\n" "${STATE}" >"${STATEFILE}"
    fi
    echo "${TRIGGER}"
    # only needed to execute body of .state-rule
    if [ "${TOUCH}" ]; then ${TOUCH} "${TRIGGER}"; fi
  fi
elif [ ! "${DIFF}" ]; then
  >&2 echo "ERROR: please install diffutils - diff command is missing!"
  exit 1
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
