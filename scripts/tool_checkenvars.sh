#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Maintained in LIBXS and copied into dependent projects by "make documentation".
# Edit it in LIBXS: a change made in a copy is overwritten.
#
# Lists the environment variables the source reads, and checks that the ones
# carrying the project's prefix are documented: the policy calls a variable
# that this script reports but the documentation does not mention a defect.
# Foreign variables (OMP_*, PMI_*, NEO*, SLURM_*) are not ours to document,
# so only the prefixed ones are checked.
#
#   tool_checkenvars.sh --list   list every variable, grouped by prefix
#   tool_checkenvars.sh          report undocumented prefixed variables
#   tool_checkenvars.sh --all    report the deferred ones as well
#
# The deferred names are not here but in tool_checkenvars.todo next to this
# script: they are per-project state, and "make documentation" copies this
# script from LIBXS, which would revert a name removed from a copy. The list
# is a to-do, not a permission, so a name that is documented by now fails the
# check as well, asking to be dropped.
TODO="tool_checkenvars.todo"

FIND=$(command -v find)
SORT=$(command -v sort)
SED=$(command -v gsed)
GIT=$(command -v git)
RESULT=0

# GNU sed is desired (macOS)
if [ ! "${SED}" ]; then
  SED=$(command -v sed)
fi

# Caption underlined and overlined by a rule of its own width.
banner() {
  echo "$1" | ${SED} "s/./=/g"
  echo "$1"
  echo "$1" | ${SED} "s/./=/g"
}

HERE="$(cd "$(dirname "$0")" && pwd -P)"
SRC="${HERE}/../src"
EXT="c"

# Amalgamated layout: the sources sit next to the headers.
if [ ! -d "${SRC}" ]; then
  for DIR in "${HERE}/../libxs" "${HERE}/../libxstream"; do
    if [ -d "${DIR}" ]; then
      SRC="${DIR}"
      break
    fi
  done
fi

if [ ! "${FIND}" ] || [ ! "${SORT}" ] || [ ! "${SED}" ] || [ ! -d "${SRC}" ]; then
  >&2 echo "ERROR: missing prerequisites!"
  RESULT=1
else
  export LC_ALL=C
  ENVARS="$(${FIND} "${SRC}" -type f -name "*.${EXT}" -exec \
    "${SED}" "s/getenv[[:space:]]*([[:space:]]*\".[^\"]*/\n&/g" {} \; | \
     ${SED} -n "s/.*getenv[[:space:]]*([[:space:]]*\"\(.[^\"]*\)..*/\1/p" | \
     ${SORT} -u)"
  PREFIXED="$(echo "${ENVARS}" | ${SED} -n "/^LIBXS\(TREAM\)\?_/p")"
  if [ "--list" = "$1" ]; then
    # The caption names the prefix the source actually uses.
    PROJECT="$(echo "${PREFIXED}" | ${SED} -n "1s/^\(LIBXS\(TREAM\)\?\)_.*/\1/p")"
    banner "Other environment variables"
    echo "${ENVARS}" | ${SED} "/^LIBXS\(TREAM\)\?_/d"
    banner "${PROJECT:-Project} environment variables"
    echo "${PREFIXED}"
  elif [ ! "${GIT}" ]; then
    >&2 echo "ERROR: missing prerequisites!"
    RESULT=1
  elif ! cd "${HERE}/.."; then
    >&2 echo "ERROR: cannot enter the repository!"
    RESULT=1
  else
    MISSING=""
    STALE=""
    UNDOC=""
    DEFERRED=" $(${SED} "s/#.*//" "${HERE}/${TODO}" 2>/dev/null | tr '\n' ' ' \
      | tr -s ' ') "
    for VAR in ${PREFIXED}; do
      if ! ${GIT} grep -qwF "${VAR}" -- "*.md"; then
        UNDOC="${UNDOC} ${VAR} "
        case "${DEFERRED}" in
        *" ${VAR} "*) [ "--all" = "$1" ] && MISSING="${MISSING} ${VAR}" ;;
        *) MISSING="${MISSING} ${VAR}" ;;
        esac
      fi
    done
    for VAR in ${DEFERRED}; do
      case " ${UNDOC}" in
      *" ${VAR} "*) ;;
      *) STALE="${STALE} ${VAR}" ;;
      esac
    done
    if [ "${MISSING}" ]; then
      >&2 echo "ERROR: undocumented environment variables:${MISSING}"
      RESULT=1
    fi
    if [ "${STALE}" ]; then
      # Lowering the list is always correct, so it happens here rather than
      # being asked for. Adding never does: a new undocumented variable has
      # to fail. The run fails either way, to be reviewed and repeated.
      SCRIPT=""
      for VAR in ${STALE}; do
        SCRIPT="${SCRIPT}/^[[:space:]]*${VAR}[[:space:]]*\$/d;"
      done
      if ${SED} "${SCRIPT}" "${HERE}/${TODO}" >"${HERE}/${TODO}.tmp"; then
        mv "${HERE}/${TODO}.tmp" "${HERE}/${TODO}"
        >&2 echo "ERROR: documented by now, dropped from ${TODO}:${STALE}"
      else
        rm -f "${HERE}/${TODO}.tmp"
        >&2 echo "ERROR: documented by now, drop from ${TODO}:${STALE}"
      fi
      RESULT=1
    fi
  fi
fi

exit ${RESULT}
