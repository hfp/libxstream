#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxstream/                     #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
SED=$(command -v gsed)
LS=$(command -v ls)
MV=$(command -v mv)
RM=$(command -v rm)
WC=$(command -v wc)

# GNU sed is desired (macOS)
if [ ! "${SED}" ]; then
  SED=$(command -v sed)
fi

# device-tag of a CSV-file, i.e. the bracketed id of its first entry
csvtag() {
  ${SED} -n "2s/^[^;]*\[\(0[xX][0-9a-fA-F]*\)\].*/\1/p" "$1" 2>/dev/null
}

# LIBXS's predict_params.x (built on demand) from LIBXSROOT, a sibling, or HOME
predictx() {
  for ROOT in "${LIBXSROOT}" "$1/../../../libxs" "${HOME}/libxs"; do
    if [ "${ROOT}" ] && [ -d "${ROOT}/samples/predict" ]; then
      if [ ! -x "${ROOT}/samples/predict/predict_params.x" ] && [ "$(command -v make)" ]; then
        make -C "${ROOT}/samples/predict" predict_params.x >/dev/null 2>&1
      fi
      if [ -x "${ROOT}/samples/predict/predict_params.x" ]; then
        echo "${ROOT}/samples/predict/predict_params.x"
        break
      fi
    fi
  done
}

if [ "${SED}" ] && [ "${LS}" ] && [ "${MV}" ] && [ "${RM}" ] && [ "${WC}" ]; then
  while test $# -gt 0; do
    case "$1" in
    -h|--help)
      HELP=1
      shift $#;;
    -f|--force)
      FORCE=1
      shift 1;;
    -e|--expand)
      EXPAND=1
      shift 1;;
    *)
      break;;
    esac
  done
  HERE=$(cd "$(dirname "$0")" && pwd -P)
  PARAMDIR="${HERE}/params"
  # merged CSV-file, beside the script rather than in the caller's directory
  CSVFILE="${HERE}/tune_multiply.csv"
  STAMP="${HERE}/.smm_params.stamp"
  # JSON-files are written aside, i.e. never into a corpus (file dates)
  TMPDIR="${HERE}/tmp"
  MATCH=${1:-*}
  if [ "${HELP}" ]; then
    echo "Usage: $0 [-f|--force] [-e|--expand] [pattern]"
    echo "       Merges each private params/<device>-0x<id> directory"
    echo "       into the published params/tune_multiply_<device>.csv"
    echo "       and rebuilds its prediction model (.bin) using LIBXS"
    echo "       (LIBXSROOT, a sibling libxs, or \$HOME/libxs)"
    echo "       -e|--expand: reverse, i.e. write the JSON-files of each"
    echo "        CSV-file to tmp/<device>-0x<id> (no corpus is touched)"
    echo "       -f|--force: write a CSV-file that does not exist yet,"
    echo "        or expand into a directory that is not empty"
    echo "       pattern: limit to matching devices"
    exit 0
  fi
  RESULT=0
  NDEVICES=0
  if [ "${EXPAND}" ]; then
    # shellcheck disable=SC2231
    for CSV in "${PARAMDIR}/tune_multiply_"${MATCH}.csv; do
      if [ ! -f "${CSV}" ]; then continue; fi
      NDEVICES=$((NDEVICES+1))
      NAME=${CSV##*/}
      NAME=${NAME%.csv}
      DEVICE=${NAME#tune_multiply_}
      TAG=$(csvtag "${CSV}")
      echo
      if [ ! "${TAG}" ]; then
        >&2 echo "ERROR: ${NAME}.csv carries no device id!"
        RESULT=1
        continue
      fi
      DEST="${TMPDIR}/${DEVICE}-${TAG}"
      if [ -d "${DEST}" ] && [ ! "${FORCE}" ] && \
         [ "0" != "$(${LS} -1 "${DEST}" | ${WC} -l)" ];
      then
        >&2 echo "ERROR: ${DEST#"${HERE}/"} is not empty (needs -f)!"
        RESULT=1
        continue
      fi
      echo "Expanding ${NAME}.csv..."
      if ! "${HERE}/tune_multiply.py" -M "${CSV}" -p "${DEST}"; then
        RESULT=1
      fi
    done
  else
    PREDICTX=$(predictx "${HERE}")
    # MATCH is a pattern rather than a name
    # shellcheck disable=SC2231
    for DIR in "${PARAMDIR}/"${MATCH}-0[xX]*/; do
      if [ ! -d "${DIR}" ]; then continue; fi
      NDEVICES=$((NDEVICES+1))
      NAME=${DIR%/}
      NAME=${NAME##*/}
      DEVICE=${NAME%-0[xX]*}
      TAG=${NAME##*-}
      TARGET="${PARAMDIR}/tune_multiply_${DEVICE}.csv"
      echo
      echo "Merging ${NAME}..."
      : >"${STAMP}"
      "${HERE}/tune_multiply.py" -p "${DIR}" -m2 -o "${CSVFILE}"
      # a merge without parameters leaves an earlier CSV-file behind
      if [ ! "${CSVFILE}" -nt "${STAMP}" ]; then
        >&2 echo "ERROR: ${NAME} produced no parameters!"
        RESULT=1
        continue
      fi
      CSVTAG=$(csvtag "${CSVFILE}")
      if [ "${CSVTAG}" ] && [ "${TAG}" != "${CSVTAG}" ]; then
        >&2 echo "ERROR: ${NAME} holds device ${CSVTAG} rather than ${TAG}!"
        RESULT=1
        continue
      fi
      if [ -e "${TARGET}" ]; then
        OLDTAG=$(csvtag "${TARGET}")
        if [ "${OLDTAG}" ] && [ "${CSVTAG}" ] && [ "${OLDTAG}" != "${CSVTAG}" ]; then
          >&2 echo "ERROR: ${TARGET##*/} holds device ${OLDTAG} rather than ${CSVTAG}!"
          RESULT=1
          continue
        fi
      elif [ ! "${FORCE}" ]; then
        >&2 echo "ERROR: no ${TARGET##*/} to update (new device needs -f)!"
        RESULT=1
        continue
      fi
      NKERNELS=$(($(${WC} -l <"${CSVFILE}")-1))
      ${MV} -f "${CSVFILE}" "${TARGET}"
      echo "Wrote ${NKERNELS} kernel(s) to ${TARGET##*/}."
      # the build embeds the model in place of the CSV-file, hence it must follow
      MODEL="${TARGET%.csv}.bin"
      if [ "${PREDICTX}" ]; then
        if "${PREDICTX}" rf trees8 "${TARGET}" "${MODEL}" >/dev/null; then
          echo "Wrote model ${MODEL##*/}."
        else
          >&2 echo "ERROR: failed to build ${MODEL##*/}!"
          RESULT=1
        fi
      elif [ -e "${MODEL}" ]; then
        >&2 echo "ERROR: ${MODEL##*/} is stale (LIBXS predict_params.x not found)!"
        RESULT=1
      fi
    done
    # a device that failed late leaves its merged CSV-file behind
    if [ "${CSVFILE}" -nt "${STAMP}" ]; then ${RM} -f "${CSVFILE}"; fi
    ${RM} -f "${STAMP}"
  fi
  if [ "0" = "${NDEVICES}" ]; then
    >&2 echo "ERROR: no device matches ${MATCH}!"
    RESULT=1
  fi
  exit ${RESULT}
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
