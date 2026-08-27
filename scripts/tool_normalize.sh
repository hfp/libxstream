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
# Front-end for the rules in .pre-commit-config.yaml, which continuous
# integration runs as well.
#
#   tool_normalize.sh --install   install the Git hook (once per clone)
#   tool_normalize.sh [DIR]       check and fix the tree, or just DIR
#
HERE=$(cd "$(dirname "$0")" && pwd -P)
PRECOMMIT=$(command -v pre-commit)
GIT=$(command -v git)
RESULT=0

if [ ! "${PRECOMMIT}" ] || [ ! "${GIT}" ]; then
  >&2 echo "ERROR: pre-commit is missing (pip install --user pre-commit)!"
  RESULT=1
elif ! cd "${HERE}/.."; then
  >&2 echo "ERROR: cannot enter the repository!"
  RESULT=1
elif [ "--install" = "$1" ]; then
  ${PRECOMMIT} install
  RESULT=$?
elif [ "$1" ]; then
  # shellcheck disable=SC2046
  ${PRECOMMIT} run --show-diff-on-failure --files $(${GIT} ls-files "$1")
  RESULT=$?
else
  ${PRECOMMIT} run --show-diff-on-failure --all-files
  RESULT=$?
fi

exit ${RESULT}
