#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# shellcheck disable=SC2086
#
HERE=$(cd "$(dirname "$0")" && pwd -P)
LIBS=${HERE}/../lib

PROJECT=libxstream
# symbols a library carries without being part of the ABI (a dependency linked in, or names another API prescribes)
FOREIGN="libxs clGetExtensionFunctionAddress clIcdGetPlatformIDsKHR"
#EXCLUDE="libxstreamf"
INCLUDE="*"
ABINEW=.abi.log
ABITMP=.abi.tmp
ABICUR=.abi.txt

BASENAME=$(command -v basename)
SORT=$(command -v sort)
DIFF=$(command -v diff)
SED=$(command -v gsed)
CUT=$(command -v cut)
LS=$(command -v ls)
TR=$(command -v tr)
WC=$(command -v wc)
CP=$(command -v cp)
MV=$(command -v mv)
NM=$(command -v nm)
REALPATH=$(command -v realpath)

# GNU sed is desired (macOS)
if [ ! "${SED}" ]; then
  SED=$(command -v sed)
fi
# shorter paths in what is printed (GNU realpath; others keep the path as is)
if [ "${REALPATH}" ]; then
  LIBS=$(${REALPATH} -s --relative-to=. "${LIBS}" 2>/dev/null || echo "${LIBS}")
fi

# a pattern matching several files is not a single operand of "test"
count() {
  ${LS} -1 "$@" 2>/dev/null | ${WC} -l
}

# a dependency's symbols, including what its Fortran compiler generated
foreign() {
  for NAME in ${FOREIGN}; do
    case "$1" in
      "${NAME}" | "${NAME}_"* | "${NAME}."* | "__${NAME}_MOD_"*) return 0 ;;
    esac
  done
  return 1
}

if [ "${NM}" ] && [ "${SED}"  ] && [ "${CUT}"  ] && \
   [ "${LS}" ] && [ "${CP}"   ] && [ "${MV}"   ] && \
   [ "${TR}" ] && [ "${WC}" ] && [ "${SORT}" ] && \
   [ "${DIFF}" ] && [ "${BASENAME}" ];
then
  PROJUPP=$(${TR} '[:lower:]' '[:upper:]' <<<"${PROJECT}")
  if [ "0" != "$(count "${LIBS}"/${INCLUDE}.so)" ]; then
    LIBARGS="-D"
    LIBTYPE=so
  elif [ "0" != "$(count "${LIBS}"/${INCLUDE}.a)" ]; then
    LIBTYPE=a
  else
    >&2 echo "ERROR: no ${PROJUPP} library found in ${LIBS}, see \"make STATIC=0 SYM=1\"."
    exit 1
  fi
  # determine behavior of sort command
  export LC_ALL=C
  ${CP} /dev/null ${ABINEW}
  for LIBFILE in "${LIBS}"/${INCLUDE}."${LIBTYPE}"; do
    LIB=$(${BASENAME} "${LIBFILE}" .${LIBTYPE})
    if [ ! "${EXCLUDE}" ] || [ "$(${SED} "/\b${LIB}\b/d" <<<"${EXCLUDE}")" ]; then
      if [ ! "${CMD}" ]; then # try certain flags only once
        if ${NM} --defined-only -p ${LIBARGS} "${LIBFILE}" >/dev/null 2>/dev/null; then
          CMD="${NM} --defined-only --no-sort ${LIBARGS}"
        else
          CMD="${NM} ${LIBARGS}"
        fi
      fi
      echo "Checking ${LIB}..."
      while IFS= read -r LINE; do
        SYMBOL=$(${SED} -n "/ T /p" <<<"${LINE}" | ${CUT} -d" " -f3)
        if [ "${SYMBOL}" ] && ! foreign "${SYMBOL}"; then
          # cleanup compiler-specific symbols (Intel Fortran, GNU Fortran)
          SYMBOL=$(${SED} <<<"${SYMBOL}" \
            -e "s/^${PROJECT}_mp_${PROJECT}_\(..*\)_/${PROJECT}_\1/" \
            -e "s/^__${PROJECT}_MOD_${PROJECT}_/${PROJECT}_/")
          if [ "$(${SED} -n "/^${PROJUPP}_[^.]/p" <<<"${SYMBOL}")" ] || \
             [ "$(${SED} -n "/^${PROJECT}_[^.]/p" <<<"${SYMBOL}")" ] || \
             [ "$(${SED} -n "/^c_dbcsr_acc_[^.]/p" <<<"${SYMBOL}")" ] || \
             [ "$(${SED} -n "/^offload[^.]/p" <<<"${SYMBOL}")" ];
          then
            echo "${SYMBOL}" >>${ABINEW}
          elif [ ! "$(${SED} <<<"${SYMBOL}" -n "/^__${PROJECT}_MOD___/p")" ] && \
               [ ! "$(${SED} <<<"${SYMBOL}" -n "/^${PROJECT}\._/p")" ] && \
               [ ! "$(${SED} <<<"${SYMBOL}" -n "/^__wrap_..*/p")" ] && \
               [ ! "$(${SED} <<<"${SYMBOL}" -n "/^internal_/p")" ] && \
               [ ! "$(${SED} <<<"${SYMBOL}" -n "/^_init/p")" ] && \
               [ ! "$(${SED} <<<"${SYMBOL}" -n "/^_fini/p")" ];
          then
            >&2 echo "ERROR: non-conforming function name"
            echo "${LIB} ->${SYMBOL}"
            exit 1
          fi
        fi
      done < <(eval "${CMD} ${LIBFILE}" 2>/dev/null)
    else
      echo "Excluded ${LIB}"
    fi
  done
  ${SORT} -u ${ABINEW} >${ABITMP}
  ${MV} ${ABITMP} ${ABINEW}
  if [ "so" != "${LIBTYPE}" ]; then
    echo "Note: ${PROJUPP} must be built with \"make STATIC=0 SYM|DBG=1\"!"
  fi
  REMOVED=$(${DIFF} --new-line-format="" --unchanged-line-format="" <(${SORT} ${ABICUR}) ${ABINEW})
  if [ ! "${REMOVED}" ]; then
    ${CP} ${ABINEW} ${ABICUR}
    echo "Successfully completed."
  else
    >&2 echo "ERROR: removed or renamed function(s)"
    echo "${REMOVED}"
    exit 1
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
