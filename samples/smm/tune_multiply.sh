#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxstream/                     #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
XARGS=$(command -v xargs)
SORT=$(command -v sort)
TAIL=$(command -v tail)
SED=$(command -v gsed)
TAC=$(command -v tac)
CUT=$(command -v cut)
LS=$(command -v ls)
MV=$(command -v mv)
RM=$(command -v rm)
WC=$(command -v wc)
TR=$(command -v tr)

# initial delay before auto-tuning (interactive)
WAIT_DEFAULT=12
# first line of a generated plan, which marks it as resumable
PLANTAG="tune_multiply plan"

MPI_SIZE=${PMI_SIZE:-${OMPI_COMM_WORLD_SIZE:-${PMIX_SIZE:-${SLURM_NTASKS:-1}}}}
MPI_RANK=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-${SLURM_PROCID:-0}}}}
MPI_LOCAL_RANK=${MPI_LOCALRANKID:-${OMPI_COMM_WORLD_LOCAL_RANK:-${PMI_LOCAL_RANK:-${PMIX_LOCAL_RANK:-${SLURM_LOCALID}}}}}
if [ "${MPI_LOCAL_RANK}" ] && [ ! "${MPI_LOCALRANKID}" ]; then
  export MPI_LOCALRANKID=${MPI_LOCAL_RANK}
fi

# GNU sed is desired (macOS)
if [ ! "${SED}" ]; then
  SED=$(command -v sed)
fi
# BSD/macOS has no tac, and GNU tail has no -r
if [ ! "${TAC}" ] && [ "${TAIL}" ] && ${TAIL} -r </dev/null 2>/dev/null; then
  TAC="${TAIL} -r"
fi

plan_print() {
  echo "# ${PLANTAG} of ${JSONDIR}: ${NTRIPLETS} kernel(s), ${PLANORD}"
  echo "# tuning marks a line \"#done\" and resuming skips it (see -f)"
  for MNK in ${MNKS}; do
    # the JSON that dated this line, i.e. the youngest of the kernel
    JSON=$(${LS} -1t "${JSONDIR}"/tune_multiply-*-"${MNK}"-*.json 2>/dev/null | ${SED} -n "1p")
    echo "${MNK} # ${JSON##*/}"
  done
}

if [ "${XARGS}" ] && [ "${SORT}" ] && [ "${SED}" ] && \
   [ "${LS}" ] && [ "${RM}" ] && [ "${WC}" ];
then
  EXTRA=""
  while test $# -gt 0; do
    case "$1" in
    -h|--help)
      HELP=1
      shift $#;;
    -c|--continue)
      CONTINUE=1
      shift 1;;
    -w|--wait)
      WAIT=$2
      shift 2;;
    -u|--update)
      UPDATE=1
      shift 1;;
    --plan)
      PLAN=1
      case "$2" in
      ""|-*) ;;
      *) PLANFILE=$2; shift 1;;
      esac
      shift 1;;
    -d|--delete)
      DELETE=1
      shift 1;;
    --prefer)
      PREFER=$2
      shift 2;;
    -a|--tuning-level)
      TLEVEL=$2
      shift 2;;
    -b|--backwards)
      REVERSE=1
      shift 1;;
    -t|--maxtime)
      MAXTIME=$2
      shift 2;;
    -p|--jsondir)
      JSONDIR=$2
      shift 2;;
    -k|--specid)
      SPECID=$2
      shift 2;;
    -m|--limit)
      MAXEXT=$2
      shift 2;;
    -n|--triplets)
      MAXNUM=$2
      shift 2;;
    -r|--bound)
      BOUNDL=$2
      BOUNDU=$3
      shift 3;;
    -i|--part)
      PART=$2
      shift 2;;
    -j|--nparts)
      NPARTS=$2
      shift 2;;
    -s|--batchsize)
      BATCHSIZE=$2
      shift 2;;
    -f|--file)
      MNKFILE=$2
      shift 2;;
    *)
      if [ "-" != "${1:0:1}" ]; then
        break
      else
        EXTRA+=" $1"
        shift
      fi;;
    esac
  done
  # default/basic settings
  if [ ! "${BATCHSIZE}" ]; then BATCHSIZE=0; fi
  if [ ! "${JSONDIR}" ]; then JSONDIR=.; fi
  if [ ! "${TLEVEL}" ]; then TLEVEL=-1; fi
  if [ ! "${PLANORD}" ]; then PLANORD="oldest first"; fi
  if [ ! "${NPARTS}" ]; then NPARTS=${MPI_SIZE}; fi
  if [ "0" != "$((1>NPARTS))" ]; then
    >&2 echo "ERROR: number of parts must be positive!"
    exit 1
  fi
  if [ ! "${PART}" ]; then
    PART=$((MPI_RANK%NPARTS+1))
  fi
  if [ ! "${WAIT}" ] && [ "1" != "${NPARTS}" ]; then WAIT=0; fi
  # writing a plan is not a tuning session
  if [ ! "${WAIT}" ] && [ "${PLAN}" ]; then WAIT=0; fi
  # sanity checks
  if [ "0" != "$((NPARTS<PART))" ]; then
    >&2 echo "ERROR: part-number ${PART} is larger than the requested ${NPARTS} parts!"
    exit 1
  elif [ "0" != "$((1>PART))" ]; then
    >&2 echo "ERROR: part-number must be 1-based!"
    exit 1
  fi
  if [ "${PREFER}" ]; then
    case "${PREFER}" in
    fast|new) PREFER=" --prefer=${PREFER}";;
    *)
      >&2 echo "ERROR: --prefer takes fast or new!"
      exit 1;;
    esac
  fi
  if [ "${SPECID}" ] && [ "$1" ]; then
    >&2 echo "ERROR: --specid and <triplet-spec> are mutual exclusive!"
    exit 1
  fi
  if [ "${PLAN}" ] && { [ "${UPDATE}" ] || [ "${MNKFILE}" ] || \
     [ "${SPECID}" ] || [ "$1" ]; };
  then
    >&2 echo "ERROR: --plan takes its kernels from the JSON-directory (see -p)!"
    exit 1
  fi
  if [ "${MNKFILE}" ] && [ "${SPECID}" ]; then
    >&2 echo "ERROR: --file and --specid are mutual exclusive!"
    exit 1
  fi
  # how to print standard vs error messages
  if [ ! "${HELP}" ] || [ "0" = "${HELP}" ]; then
    JSONS=$(${LS} -1t "${JSONDIR}"/tune_multiply-*-*x*x*-*.json 2>/dev/null)
    HERE=$(cd "$(dirname "$0")" && pwd -P)
    ECHO=">&2 echo"
    # acc_triplets.sh is a sibling here, but two levels up in dependent projects
    TRIPLETS="${HERE}/acc_triplets.sh"
    if [ ! -e "${TRIPLETS}" ]; then TRIPLETS="${HERE}/../../acc_triplets.sh"; fi
    MNKPAT="s/.*tune_multiply-[^-]*-\([0-9]*x[0-9]*x[0-9]*\)-.*\.json$/\1/p"
    if [ "${PLAN}" ]; then
      # JSONS is youngest first, hence prepending dates a kernel by its
      # youngest JSON and orders the kernels by that date (oldest first)
      for MNK in $(${SED} -n "${MNKPAT}" <<<"${JSONS}"); do
        case " ${MNKS} " in *" ${MNK} "*) continue;; esac
        MNKS="${MNK} ${MNKS}"
      done
    elif [ "${UPDATE}" ] && [ "0" != "${UPDATE}" ]; then
      MNKS=$(${SED} -n "${MNKPAT}" <<<"${JSONS}" \
         | ${SORT} -u -n -tx -k1,1 -k2,2 -k3,3)
    elif [ "${MNKFILE}" ]; then
      if [ ! -f "${MNKFILE}" ]; then
        >&2 echo "ERROR: file not found: ${MNKFILE}"
        exit 1
      fi
      MNKS=$(${SED} -e "s/#.*//" -e "s/[[:space:]]//g" "${MNKFILE}" \
           | ${SED} -n "/[0-9].*x.*[0-9]/p" | ${XARGS})
      # progress is recorded in a generated plan, never in a handwritten list
      if [ "${MV}" ] && [ "1" = "${NPARTS}" ] && \
         [ "$(${SED} -n "1s/^#[[:space:]]*${PLANTAG}.*/1/p" "${MNKFILE}")" ];
      then
        NDONE=$(${SED} -n "/^#done/p" "${MNKFILE}" | ${WC} -l)
        RESUME=1
      fi
    elif [ "${SPECID}" ]; then
      if [ ! -e "${TRIPLETS}" ]; then
        >&2 echo "ERROR: file not found: ${TRIPLETS}"
        exit 1
      fi
      MNKS=$(eval "${TRIPLETS} -k ${SPECID} 2>/dev/null")
    else
      if [[ "$*" != *"x"* ]]; then
        if [ ! -e "${TRIPLETS}" ]; then
          >&2 echo "ERROR: file not found: ${TRIPLETS}"
          exit 1
        fi
        MNKS=$(eval "${TRIPLETS} $* 2>/dev/null")
      else
        MNKS="$*"
      fi
    fi
  else
    ECHO="echo"
  fi
  if [ ! "${WAIT}" ] || [[ ("${HELP}" && "0" != "${HELP}") ]]; then
    eval "${ECHO} \"Usage: $0 [options] [<triplet-spec>]\""
    eval "${ECHO} \"       Options must precede triplet specification\""
    eval "${ECHO} \"       Other options go to tune_multiply.py (--opt=value)\""
    eval "${ECHO} \"       -w|--wait N: initial delay before auto-tuning (default: ${WAIT_DEFAULT} s)\""
    eval "${ECHO} \"       -c|--continue: proceed with plan if tuning is interrupted\""
    eval "${ECHO} \"       -u|--update: retune all JSONs found in directory (see -p)\""
    eval "${ECHO} \"       --plan F: write plan of JSONs by age to F or stdout,\""
    eval "${ECHO} \"        oldest kernel first (see -p, -b, and -n)\""
    eval "${ECHO} \"       -f|--file F: read MxNxK list from file (one per line, # comments)\""
    eval "${ECHO} \"        a plan is resumed, i.e. tuned kernels are skipped\""
    eval "${ECHO} \"       -s|--batchsize N: Number of batched SMMs (a.k.a. stacksize)\""
    eval "${ECHO} \"       -a|--tuning-level N=0..4: all, most, some, few, least\""
    eval "${ECHO} \"        tunables, where the default (-1) matches level 2\""
    eval "${ECHO} \"       -b|--backwards: tune in descending order of triplets\""
    eval "${ECHO} \"        (reverses before -n limits and before parts are cut)\""
    eval "${ECHO} \"       -d|--delete: delete losing duplicates when merging JSONs\""
    eval "${ECHO} \"       --prefer fast|new: duplicate winner (default: fast)\""
    eval "${ECHO} \"       -t|--maxtime N: number of seconds spent per kernel\""
    eval "${ECHO} \"       -p|--jsondir P: path to JSON-files (tuned params)\""
    eval "${ECHO} \"       -i|--part N (1-based): Nth session out of nparts\""
    eval "${ECHO} \"       -j|--nparts N: number of total sessions (see -i)\""
    eval "${ECHO} \"       -r|--bound L U: limit L**3 < MNK <= U**3\""
    eval "${ECHO} \"       -m|--limit N: limit any shape extent to N\""
    eval "${ECHO} \"       -n|--triplets N: limit number of triplet\""
    eval "${ECHO} \"       -k|--specid N: predefined triplets\""
    eval "${ECHO} \"        0-10: older to newer (larger), e.g.,\""
    eval "${ECHO} \"           0:  201 kernels\""
    eval "${ECHO} \"          10: 1266 kernels\""
    eval "${ECHO} \"       <triplet-spec>, e.g., 134 kernels \\\"23, 5 32 13 24 26, 4 9\\\"\""
    eval "${ECHO} \"         MxNxK's can be also given directly, e.g.,\""
    eval "${ECHO} \"         1x1x1 2x2x2 2x2x3 2x3x2 2x3x3 3x2x2 3x2x3 3x3x2 3x3x3\""
    eval "${ECHO} \"         (which is equivalent to \\\"1, 2 3\\\")\""
    eval "${ECHO}"
    if [ "${HELP}" ] && [ "0" != "${HELP}" ]; then exit 0; fi
  fi
  if [ "${MNKS}" ]; then
    if [ "${BOUNDL}" ] || [ "${BOUNDU}" ]; then
      if [ ! "${BOUNDL}" ]; then BOUNDL=0; elif [ ! "${BOUNDU}" ]; then BOUNDU=0; fi
      if [ "0" != "$((0<=BOUNDL))" ]; then
        for MNK in $(${SED} "s/x/*/g" <<<"${MNKS}"); do
          S=$((MNK))
          if [ "0" != "$((BOUNDL<BOUNDU))" ]; then
            if [ "0" != "$((BOUNDL**3<S&&S<=BOUNDU**3))" ]; then TMP="${TMP} ${MNK}"; fi
          else
            if [ "0" != "$((BOUNDL**3<S))" ]; then TMP="${TMP} ${MNK}"; fi
          fi
        done
        MNKS=$(${SED} "s/*/x/g" <<<"${TMP}")
      fi
    fi
    if [ "${MNKS}" ] && [ "${MAXEXT}" ] && [ "0" != "$((0<MAXEXT))" ]; then
      TMP=""
      for MNK in ${MNKS}; do
        for EXT in $(${SED} "s/x/ /g" <<<"${MNK}"); do
          if [ "0" != "$((MAXEXT<EXT))" ]; then continue 2; fi
        done
        TMP="${TMP} ${MNK}"
      done
      MNKS=${TMP}
    fi
    if [ "${REVERSE}" ] && [ "0" != "${REVERSE}" ]; then
      if [ "${TR}" ] && [ "${TAC}" ]; then
        MNKS=$(${TR} ' ' '\n' <<<"${MNKS}" | ${TAC} | ${TR} '\n' ' '; echo)
        PLANORD="newest first"
      else # tuning ascending without a word would mislabel the session
        >&2 echo "WARNING: ascending order (reversal needs tr and tac/tail)!"
      fi
    fi
    if [ "${MNKS}" ] && [ "${MAXNUM}" ] && [ "0" != "$((0<MAXNUM))" ]; then
      # a reader that quits early makes xargs report SIGPIPE, hence sed
      MNKS=$(${XARGS} -n1 <<<"${MNKS}" | ${SED} -n "1,${MAXNUM}p" | ${XARGS})
    else
      MNKS=$(${XARGS} <<<"${MNKS}")
    fi
  fi
  NTRIPLETS=$(${WC} -w <<<"${MNKS}")
  if [ "0" != "$((0==NTRIPLETS))" ]; then
    if [ "${RESUME}" ]; then
      echo "Plan ${MNKFILE} is complete (${NDONE} kernel(s) tuned)."
      exit 0
    fi
    if [ "${HELP}" ] || [ "0" = "${HELP}" ]; then exit 0; fi
    >&2 echo "ERROR: invalid or no <triplet-spec> given!"
    exit 1
  fi
  if [ "${PLAN}" ]; then
    if [ "${PLANFILE}" ]; then
      if plan_print >"${PLANFILE}"; then
        echo "Wrote plan of ${NTRIPLETS} kernel(s) to ${PLANFILE}."
      else
        >&2 echo "ERROR: cannot write plan to ${PLANFILE}!"
        exit 1
      fi
    else
      plan_print
    fi
    exit 0
  fi
  if [ "${RESUME}" ] && [ "0" != "${NDONE}" ]; then
    echo "Resuming ${MNKFILE} with ${NDONE} kernel(s) already tuned."
  fi
  if [ ! "${WAIT}" ] || [ "0" != "${WAIT}" ]; then
    if [ "0" = "$((NPARTS<=NTRIPLETS))" ]; then
      >&2 echo "WARNING: problem is over-decomposed!"
    fi
    if [ "${MPI_LOCAL_RANK}" ]; then
      echo "Session ${PART} of ${NPARTS} part(s), local MPI rank ${MPI_LOCAL_RANK}."
    else
      echo "Session ${PART} of ${NPARTS} part(s)."
    fi
  fi
  if [ ! "${MAXTIME}" ] && [[ (! "${CONTINUE}"  || \
      "${CONTINUE}" = "false"                   || \
      "${CONTINUE}" = "no"                      || \
      "${CONTINUE}" = "0") ]];
  then
    MAXTIME=160
  fi
  PARTLOSZ=$((NPARTS<NTRIPLETS?(NTRIPLETS/NPARTS):1))
  PARTUPSZ=$(((NTRIPLETS+NPARTS-1)/NPARTS))
  PARTUPNM=$((PARTUPSZ!=PARTLOSZ?(NTRIPLETS-PARTLOSZ*NPARTS):(NTRIPLETS/PARTUPSZ)))
  PARTZERO=$((PART-1))
  PARTOFFS=$((PARTZERO<=PARTUPNM?(PARTZERO*PARTUPSZ):(PARTUPNM*PARTUPSZ+(PARTZERO-PARTUPNM)*PARTLOSZ)))
  PARTSIZE=$((PART<=PARTUPNM?PARTUPSZ:PARTLOSZ))
  if [ "${MAXTIME}" ] && [ "0" != "$((0<MAXTIME))" ]; then
    if [[ ! "${WAIT}" || "0" != "${WAIT}" ]] && [ "1" = "${PART}" ]; then
      HRS=$((MAXTIME*PARTSIZE/3600))
      MNS=$(((MAXTIME*PARTSIZE-HRS*3600+59)/60))
      echo "Tuning ${NTRIPLETS} kernels will take about ${HRS}h${MNS}m."
    fi
    MAXTIME="--stop-after=${MAXTIME}"
  else
    echo "Tuning ${PARTSIZE} kernels will take an unknown time (no limit given)."
  fi
  if [ "${DELETE}" ] && [ "0" != "${DELETE}" ]; then DELETE=-d; fi
  # a here-string of an empty list still counts as one line
  if [ "${JSONS}" ]; then NJSONS=$(${WC} -l <<<"${JSONS}"); else NJSONS=0; fi
  if [ "0" != "${NJSONS}" ]; then
    if [ ! "${UPDATE}" ] || [ "0" = "${UPDATE}" ]; then
      >&2 echo "Already found ${NJSONS} (unrelated?) JSON-files."
    fi
  elif [ -e tune_multiply.csv ]; then
    >&2 echo "No JSON file found but (unrelated?) tune_multiply.csv exists."
  fi
  if [ ! "${WAIT}" ]; then WAIT=${WAIT_DEFAULT}; fi
  if [ "0" != "$((0<WAIT))" ] && [ "$(command -v sleep)" ]; then
    echo
    echo "Tuning will start in ${WAIT} seconds. Hit CTRL-C to abort."
    sleep "${WAIT}"
  fi
  N=0
  HOSTNAME=$(hostname)
  MNKPART=$(${CUT} -d' ' -f $((PARTOFFS+1))-$((PARTOFFS+PARTSIZE)) <<<"${MNKS}")
  for MNK in ${MNKPART}; do
    if [ "0" != "$(((N)<PARTSIZE))" ]; then
      if [ "1" != "${NPARTS}" ] && [ "${HOSTNAME}" ]; then
        STEP="@${HOSTNAME}"
        if [ "${MPI_LOCAL_RANK}" ]; then STEP="${STEP}:${MPI_LOCAL_RANK}"; fi
      fi
      echo
      echo "[$((N+1))/${PARTSIZE}]${STEP}: auto-tuning ${MNK}-kernel..."
      # avoid mixing database of previous results into new session
      ${RM} -rf ./opentuner.db
      eval "${HERE}/tune_multiply.py ${MNK} ${DELETE}${PREFER} -p ${JSONDIR} -s ${BATCHSIZE} -a ${TLEVEL} ${MAXTIME}${EXTRA}"
      RESULT=$?
      # an interrupted or failed kernel stays in the plan
      if [ "${RESUME}" ] && [ "0" = "${RESULT}" ]; then
        if ${SED} "s/^\(${MNK}\([[:space:]].*\)*\)$/#done \1/" \
             "${MNKFILE}" >"${MNKFILE}.tmp";
        then
          ${MV} -f "${MNKFILE}.tmp" "${MNKFILE}"
        else
          ${RM} -f "${MNKFILE}.tmp"
        fi
      fi
      # environment var. CONTINUE allows to proceed with next kernel
      # even if tune_multiply.py returned non-zero exit code
      if [[ ("0" != "${RESULT}") && \
            ("${CONTINUE}" = "" \
          || "${CONTINUE}" = "0" \
          || "${CONTINUE}" = "no" \
          || "${CONTINUE}" = "false") ]];
      then
        exit ${RESULT}
      fi
    else
      break
    fi
    N=$((N+1))
  done
  if [ "${RESULT}" ]; then
    ${RM} -rf ./opentuner.db
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
