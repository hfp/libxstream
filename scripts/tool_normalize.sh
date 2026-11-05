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
# Front-end for the rules in .pre-commit-config.yaml, which continuous
# integration runs as well.
#
#   tool_normalize.sh --install   install the Git hook (once per clone)
#   tool_normalize.sh [DIR]       check and fix the tree, or just DIR
#   tool_normalize.sh --pedantic [DIR]
#                                 as above, then apply the full rule set to DIR
#                                 (default: samples and tests), which the hook
#                                 relaxes there: no to-do list, no exclusions,
#                                 and shellcheck without its backlog
#
HERE=$(cd "$(dirname "$0")" && pwd -P)
PRECOMMIT=$(command -v pre-commit)
GIT=$(command -v git)
SHELLCHECK=$(command -v shellcheck)
RESULT=0

if [ ! "${PRECOMMIT}" ] || [ ! "${GIT}" ]; then
  >&2 echo "ERROR: pre-commit is missing (pip install --user pre-commit)!"
  RESULT=1
elif [ ! "${SHELLCHECK}" ]; then
  >&2 echo "ERROR: shellcheck is missing (apt install shellcheck, or brew install shellcheck)!"
  RESULT=1
elif ! cd "${HERE}/.."; then
  >&2 echo "ERROR: cannot enter the repository!"
  RESULT=1
elif [ "--install" = "$1" ]; then
  ${PRECOMMIT} install
  RESULT=$?
elif [ "--pedantic" = "$1" ]; then
  SCOPE=${2:-samples tests}
  # shellcheck disable=SC2086,SC2046
  FILES=$(${GIT} ls-files ${SCOPE})
  if [ ! "${FILES}" ]; then
    >&2 echo "ERROR: no files under ${SCOPE}!"
    RESULT=1
  else
    # First the fixers and the relaxed rules, as for any other run.
    # shellcheck disable=SC2086
    ${PRECOMMIT} run --show-diff-on-failure --files ${FILES}
    RESULT=$?
    # Generated sources stay with their producer, as in the hook.
    SOURCES=$(echo "${FILES}" | grep -E '\.(c|h|cl)$' | grep -v 'opencl_kernels\.h$')
    SCRIPTS=$(echo "${FILES}" | grep -E '\.(sh|bash)$')
    if [ "${SOURCES}" ]; then
      # shellcheck disable=SC2086
      "${HERE}/tool_checkstruct.py" --strict ${SOURCES} || RESULT=1
    fi
    if [ "${SCRIPTS}" ]; then
      # shellcheck disable=SC2086
      ${SHELLCHECK} ${SCRIPTS} || RESULT=1
    fi
  fi
elif [ "$1" ]; then
  # shellcheck disable=SC2046
  ${PRECOMMIT} run --show-diff-on-failure --files $(${GIT} ls-files "$1")
  RESULT=$?
else
  ${PRECOMMIT} run --show-diff-on-failure --all-files
  RESULT=$?
fi

exit ${RESULT}
