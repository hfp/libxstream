#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Maintained in LIBXS and copied into dependent projects by "make policies".
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
# DEFER carries the open findings, not a permission: a name is removed from
# it as soon as the documentation mentions the variable. LIBXS owns the
# LIBXS_* entries, LIBXSTREAM the LIBXSTREAM_* ones; the list is shared
# because the script is.
DEFER="
LIBXS_DUMP_BUILD LIBXS_DUMP_FILE LIBXS_DUMP_FILES LIBXS_MALLOC_LIMIT
LIBXS_PREDICT_DECOMPOSE_FOLDS LIBXS_PREDICT_REFINE LIBXS_PREDICT_TANGENT
LIBXS_PREDICT_WINDOW_FOLDS LIBXS_SIGNAL
LIBXSTREAM_ATOMICS LIBXSTREAM_BARRIER LIBXSTREAM_BIGGRF LIBXSTREAM_CACHE
LIBXSTREAM_CPP LIBXSTREAM_CPPBIN LIBXSTREAM_CPPFLAGS LIBXSTREAM_DEBUG
LIBXSTREAM_DEVIDS LIBXSTREAM_DEVMATCH LIBXSTREAM_DEVSPLIT LIBXSTREAM_DEVTYPE
LIBXSTREAM_DUMP LIBXSTREAM_INTEL LIBXSTREAM_NCCS LIBXSTREAM_NLOCKS
LIBXSTREAM_NV LIBXSTREAM_PRIORITY LIBXSTREAM_STAGE LIBXSTREAM_STAGE_GRAIN
LIBXSTREAM_STAGE_NT LIBXSTREAM_VENDOR LIBXSTREAM_WA
"

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
    # DEFER is written one group per line, so fold it into a single line.
    DEFERRED=" $(echo "${DEFER}" | tr '\n' ' ' | tr -s ' ') "
    for VAR in ${PREFIXED}; do
      if ! ${GIT} grep -qwF "${VAR}" -- "*.md"; then
        if [ "--all" = "$1" ]; then
          MISSING="${MISSING} ${VAR}"
        else
          case "${DEFERRED}" in
          *" ${VAR} "*) ;;
          *) MISSING="${MISSING} ${VAR}" ;;
          esac
        fi
      fi
    done
    if [ "${MISSING}" ]; then
      >&2 echo "ERROR: undocumented environment variables:${MISSING}"
      RESULT=1
    fi
  fi
fi

exit ${RESULT}
