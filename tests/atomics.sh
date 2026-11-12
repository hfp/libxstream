#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Runs the SMM (fp64, fp32) and DBM (fp64) samples with each FP-atomics variant
# (detected, cmpxchg, xchg) on every GPU in GPUS, which are simulated GPUs
# (VGPU_GPU, see "vgpu list-gpus") or "local" for whatever device is present.
# What a device reports, what the test-compile of a vendor builtin said, which
# expression was chosen (LIBXSTREAM_DEBUG), and whether the result is right.
#
# On a simulated GPU, the detected variant runs once more with one work-group
# at a time (VGPU_THREADS=1), so that a compare-and-swap loop never retries and
# what the simulator counts does not move between runs. The counts give a score
# per kernel: flops / (2 x instructions x wave size), the share of lane-slots
# that would be useful if every slot were an FMA. It is idealized on purpose;
# a known time lower bound and a measured time are what it is meant to meet.
#
# The named mixes of samples/mixes are counted the same way, and every kernel
# is expressed in them (scripts/tool_score.py, if numpy and scipy are present).
#
# Into LOGS (default: ./atomics): one log per run, results.md (every run),
# counts.tsv (the counts), mixes.tsv (the same in long form, per launch, with
# the named mixes), gaps.md, and summary.md (compact table, score,
# changes against PREV, a counts.tsv of an earlier run, and what is not
# covered). META names what the counts depend on (toolchain, simulator), so
# that a change of it is told apart from a change of the code.
#
# GAP is neither a pass nor a failure: the run failed and the simulator said
# why ("VirtualGPU <layer>: ..."), or the OpenCL runtime lists no device. The
# exit status is non-zero if a run failed, but not for a GAP.
###############################################################################

HERE=$(cd "$(dirname "$0")" && pwd -P)
ROOT=${HERE}/..
LOGS=${LOGS:-${PWD}/atomics}
SHAPE=${SHAPE:-13x5x7}
STACK=${STACK:-1000}
MAKE=${MAKE:-$(command -v make)}
NPROC=$(nproc 2> /dev/null || echo 4)
KERNELS="smm_double smm_float dbm"
VARIANTS="detected cmpxchg xchg"
STATUS=0

if [ -z "${GPUS}" ]; then
  if command -v vgpu > /dev/null; then GPUS=$(vgpu list-gpus | grep '^amd/' | paste -sd' ' -); fi
  GPUS=${GPUS:-local}
fi
mkdir -p "${LOGS}" || exit 1
BIN=$(mktemp -d) || exit 1
trap 'rm -rf "${BIN}"' EXIT
RESULTS=${LOGS}/results.md
COUNTS=${LOGS}/counts.tsv
MIXES=${LOGS}/mixes.tsv
GAPS=${LOGS}/gaps.md
SUMMARY=${LOGS}/summary.md
: > "${GAPS}"
echo "| GPU | Device | Verified profile | cl_ext_float_atomics | Kernel | LIBXSTREAM_ATOMICS | Probe | ATOMIC_ADD_GLOBAL | Result |" > "${RESULTS}"
echo "|---|---|---|---|---|---|---|---|---|" >> "${RESULTS}"
{ echo "# ${META:-unknown}"
  printf 'gpu\tkernel\texpr\twave\tlaunches\tflops\tstack\tinstructions\tvalu\tmfma\tsalu\tsmem\tvmem\tflat\tlds\tflat_atomic\n'
} > "${COUNTS}"
# the same in long form, per launch and for the named mixes too: one row per count, for scripts/tool_score.py
{ echo "# ${META:-unknown}"; printf 'target\tkernel\tcategory\tvalue\n'; } > "${MIXES}"

# the benchmarks, built once (the SMM sample for each precision)
for ELEM in float double; do
  "${MAKE}" -j "${NPROC}" -C "${ROOT}/samples/smm" ELEM_TYPE=${ELEM} || exit 1
  cp "${ROOT}/samples/smm/acc_bench.x" "${BIN}/acc_bench_${ELEM}.x" || exit 1
done
"${MAKE}" -j "${NPROC}" -C "${ROOT}/samples/dbm" || exit 1
for ELEM in float double; do
  "${MAKE}" -j "${NPROC}" -C "${ROOT}/samples/mixes" ELEM_TYPE=${ELEM} || exit 1
  cp "${ROOT}/samples/mixes/mixes.x" "${BIN}/mixes_${ELEM}.x" || exit 1
done

# run <log> <gpu> <command...>: the exit code, with the environment of a run
run() {
  local LOG=$1 GPU=$2 RESULT=0
  shift 2
  if [ "local" != "${GPU}" ]; then set -- env VGPU_GPU="${GPU}" "$@"; fi
  env LIBXSTREAM_DEBUG=1 LIBXSTREAM_VERBOSE=2 DBM_MULTIPLY_SMM=-1 timeout 900 "$@" > "${LOG}" 2>&1 || RESULT=$?
  echo "${RESULT}"
}

# counted <log> <pattern>: the launches whose kernel name matches, summed
# (the launch trace names each kernel just before its counts are written)
counted() {
  awk -v pat="$2" '
    /^VirtualGPU [A-Za-z]*: launch / { name[n++] = $4 }
    /^VGPU_DISPATCH / {
      k = name[m++]
      if (k ~ pat) {
        ++launches
        for (i = 2; i <= NF; ++i) { split($i, kv, "="); if ("wave" == kv[1]) wave = kv[2]; else sum[kv[1]] += kv[2] }
      }
    }
    END {
      if (0 < launches) {
        printf "%s\t%d\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\n", wave, launches,
          sum["instructions"], sum["valu"], sum["mfma"], sum["salu"], sum["smem"], sum["vmem"],
          sum["flat"], sum["lds"], sum["flat_atomic"], sum["waves"]
      }
    }' "$1"
}

# per_launch <log> <pattern> <target> <kernel>: rows of MIXES, every count divided by the launches
per_launch() {
  awk -v pat="$2" -v target="$3" -v kernel="$4" '
    /^VirtualGPU [A-Za-z]*: launch / { name[n++] = $4 }
    /^VGPU_DISPATCH / {
      k = name[m++]
      if (k ~ pat) {
        ++launches
        for (i = 2; i <= NF; ++i) { split($i, kv, "="); if ("device" != kv[1]) sum[kv[1]] += kv[2] }
      }
    }
    END {
      if (0 < launches) {
        printf "%s\t%s\tlaunches\t%d\n", target, kernel, launches
        for (c in sum) printf "%s\t%s\t%s\t%.10g\n", target, kernel, c, sum[c] / launches
      }
    }' "$1" >> "${MIXES}"
}

declare -A OUT EXPR PROBE TARGET VERIFY
for GPU in ${GPUS}; do
  if [ "local" != "${GPU}" ]; then INFO=$(VGPU_GPU=${GPU} clinfo --raw 2> /dev/null || true)
  else INFO=$(clinfo --raw 2> /dev/null || true); fi
  DEVICE=$(sed -n 's/.*CL_DEVICE_NAME[[:space:]]*//p' <<< "${INFO}" | sed -n 1p)
  FPEXT=$(grep -q cl_ext_float_atomics <<< "${INFO}" && echo yes || echo no)
  if command -v vgpu > /dev/null && [ "local" != "${GPU}" ]; then
    VERIFIED=$(grep -q '"verified": *true' <<< "$(vgpu info --gpu "${GPU}" --json)" && echo yes || echo no)
  else VERIFIED=-; fi
  TARGET[${GPU}]=${DEVICE%%:*}
  VERIFY[${GPU}]=${VERIFIED}
  if [ -z "${DEVICE}" ]; then
    echo "| ${GPU} | none | ${VERIFIED} | - | - | - | - | - | GAP |" >> "${RESULTS}"
    echo "- ${GPU}: the OpenCL runtime lists no device" >> "${GAPS}"
    continue
  fi
  for KERNEL in ${KERNELS}; do
    case ${KERNEL} in
      smm_*) CMD="${BIN}/acc_bench_${KERNEL#smm_}.x"; PATTERN="smm[0-9]" ;;
      *) CMD="${ROOT}/samples/dbm/dbm_bench.x"; PATTERN="dbm_multiply" ;;
    esac
    for VARIANT in ${VARIANTS}; do
      LOG=${LOGS}/${GPU//\//_}-${KERNEL}-${VARIANT}.log
      if [ "detected" = "${VARIANT}" ]; then SET="-u LIBXSTREAM_ATOMICS"; else SET="LIBXSTREAM_ATOMICS=${VARIANT}"; fi
      # shellcheck disable=SC2086
      RESULT=$(run "${LOG}" "${GPU}" env ${SET} "${CMD}" 1 "${STACK}" "${SHAPE}")
      if [ "dbm" = "${KERNEL}" ] && [ "0" = "${RESULT}" ] && ! grep -q "DBM-kernel" "${LOG}"; then RESULT=host; fi
      X=$(grep -o 'ATOMIC_ADD_GLOBAL(A,B)=[^ "]*' "${LOG}" | sed 's/^ATOMIC_ADD_GLOBAL(A,B)=//' | sort -u | paste -sd' ' || true)
      if grep -q ') accepted$' "${LOG}"; then P=accepted
      elif grep -q ') not accepted' "${LOG}"; then P=rejected
      else P=-; fi
      SIM=$(grep -v 'VirtualGPU [A-Za-z]*: launch ' "${LOG}" | grep -m1 '^VirtualGPU [A-Za-z]*:' | tr -d '`|' || true)
      if [ "0" = "${RESULT}" ]; then O=ok
      elif [ -n "${SIM}" ]; then
        O=GAP
        echo "- ${GPU} ${KERNEL} ${VARIANT} (exit ${RESULT}): \`${SIM}\`" >> "${GAPS}"
      else O="FAILED (${RESULT})"; STATUS=1; fi
      OUT[${GPU},${KERNEL},${VARIANT}]=${O}
      if [ "detected" = "${VARIANT}" ]; then EXPR[${GPU},${KERNEL}]=${X}; PROBE[${GPU},${KERNEL}]=${P}; fi
      echo "| ${GPU} | ${DEVICE} | ${VERIFIED} | ${FPEXT} | ${KERNEL} | ${VARIANT} | ${P} | \`${X:-none}\` | ${O} |" >> "${RESULTS}"
      if [ "ok" != "${O}" ]; then
        echo "::group::${GPU} ${KERNEL} ${VARIANT}: ${O}"
        tail -n 40 "${LOG}"
        echo "::endgroup::"
      fi
    done
    # what the simulator counts for the detected variant, one work-group at a time
    if [ "local" != "${GPU}" ] && [ "ok" = "${OUT[${GPU},${KERNEL},detected]}" ]; then
      LOG=${LOGS}/${GPU//\//_}-${KERNEL}-count.log
      # shellcheck disable=SC2086
      RESULT=$(run "${LOG}" "${GPU}" env -u LIBXSTREAM_ATOMICS VGPU_THREADS=1 VGPU_DISPATCH_STATS=1 VGPU_TRACE_LAUNCHES=1 \
        "${CMD}" 1 "${STACK}" "${SHAPE}")
      ROW=$(counted "${LOG}" "${PATTERN}")
      # the shape the benchmark ran: "<binary> nrepeat stack m n k ..."
      read -r STK M N K <<< "$(awk '$1 ~ /bench[^ ]*\.x$/ && $2 ~ /^[0-9]+$/ { print $3, $4, $5, $6; exit }' "${LOG}")"
      if [ "0" = "${RESULT}" ] && [ -n "${ROW}" ] && [ -n "${K}" ]; then
        read -r WAVE LAUNCHES REST <<< "${ROW}"
        FLOPS=$((2 * M * N * K * STK * LAUNCHES))
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${GPU}" "${KERNEL}" "${EXPR[${GPU},${KERNEL}]:-none}" "${WAVE}" "${LAUNCHES}" \
          "${FLOPS}" "${STK}" "$(cut -f3-11 <<< "${ROW}")" >> "${COUNTS}"
        per_launch "${LOG}" "${PATTERN}" "${GPU}" "${KERNEL}"
        printf '%s\t%s\twork_flops\t%s\n' "${GPU}" "${KERNEL}" "$((2 * M * N * K * STK))" >> "${MIXES}"
      fi
    fi
  done
  # the named mixes, counted (one work-group at a time) and checked
  if [ "local" != "${GPU}" ]; then
    for ELEM in double float; do
      LOG=${LOGS}/${GPU//\//_}-mixes_${ELEM}-count.log
      RESULT=$(run "${LOG}" "${GPU}" env -u LIBXSTREAM_ATOMICS VGPU_THREADS=1 VGPU_DISPATCH_STATS=1 VGPU_TRACE_LAUNCHES=1 \
        "${BIN}/mixes_${ELEM}.x")
      SIM=$(grep -v 'VirtualGPU [A-Za-z]*: launch ' "${LOG}" | grep -m1 '^VirtualGPU [A-Za-z]*:' | tr -d '`|' || true)
      if [ "0" = "${RESULT}" ]; then
        O=ok
        for MIX in stream gemm atomics gather empty; do
          per_launch "${LOG}" "mixes_${MIX}" "${GPU}" "${MIX}_${ELEM}"
          # the work of one launch, as the sample reports it (bytes, flops, updates)
          sed -n "s/^mixes_${MIX}: .* \(bytes\|flops\|updates\)=\([0-9]*\) .*/\1 \2/p" "${LOG}" | sed -n 1p |
            while read -r WHAT VALUE; do printf '%s\t%s\twork_%s\t%s\n' "${GPU}" "${MIX}_${ELEM}" "${WHAT}" "${VALUE}"; done >> "${MIXES}"
        done
      elif [ -n "${SIM}" ]; then
        O=GAP
        echo "- ${GPU} mixes_${ELEM} (exit ${RESULT}): \`${SIM}\`" >> "${GAPS}"
      else O="FAILED (${RESULT})"; STATUS=1; fi
      echo "| ${GPU} | ${DEVICE} | ${VERIFIED} | ${FPEXT} | mixes_${ELEM} | detected (counted) | - | - | ${O} |" >> "${RESULTS}"
      if [ "ok" != "${O}" ]; then
        echo "::group::${GPU} mixes_${ELEM}: ${O}"
        tail -n 40 "${LOG}"
        echo "::endgroup::"
      fi
    done
  fi
done
# stable order of the long form (awk's "for (c in sum)" has none)
{ head -2 "${MIXES}"; tail -n +3 "${MIXES}" | sort; } > "${MIXES}.sorted" && mv "${MIXES}.sorted" "${MIXES}"

# score <gpu> <kernel>: flops / (2 x instructions x wave size) in percent, or "-"
score() {
  awk -F'\t' -v gpu="$1" -v kernel="$2" '
    $1 == gpu && $2 == kernel && 0 < $8 { printf "%.2g%%", 100 * $6 / (2 * $8 * $4); found = 1; exit }
    END { if (!found) printf "-" }' "${COUNTS}"
}

{ echo "### FP atomics"
  echo
  echo "A kernel's cell: the detected, cmpxchg and xchg variants, then the score of the detected one (higher is better, see below)."
  echo
  echo "| GPU | Target | Verified profile | Probe fp32/fp64 | ${KERNELS// / | } |"
  echo "|---|---|---|---|---|---|---|"
  for GPU in ${GPUS}; do
    LINE="| ${GPU} | ${TARGET[${GPU}]:-none} | ${VERIFY[${GPU}]} | ${PROBE[${GPU},smm_float]:--}/${PROBE[${GPU},smm_double]:--} |"
    for KERNEL in ${KERNELS}; do
      if [ -n "${TARGET[${GPU}]}" ]; then
        CELL=""
        for VARIANT in ${VARIANTS}; do CELL="${CELL}${OUT[${GPU},${KERNEL},${VARIANT}]:-GAP} "; done
        LINE="${LINE} ${CELL}($(score "${GPU}" "${KERNEL}")) |"
      else LINE="${LINE} GAP |"; fi
    done
    echo "${LINE}"
  done
  if [ -s "${GAPS}" ]; then
    echo; echo "**Gaps seen** (not tested, hence neither passed nor failed):"; cat "${GAPS}"
  fi
  echo
  echo "<details><summary>Score: what the simulator counted (detected variant, one work-group at a time)</summary>"
  echo
  echo "Score = flops / (2 x instructions x wave size), higher is better: the share of lane-slots that would be useful if every slot were an FMA. Idealized (all lanes active, every instruction one slot); matrix instructions can exceed 100%. Shares are of all instructions; atomics are instructions per stack entry."
  echo
  echo "| GPU | Kernel | ATOMIC_ADD_GLOBAL | Wave | Launches | Instructions | VALU | MFMA | Memory | LDS | Atomics | Score (higher is better) |"
  echo "|---|---|---|---|---|---|---|---|---|---|---|---|"
  awk -F'\t' '
    NR > 2 {
      pct = (0 < $8 ? 100 / $8 : 0)
      expr = $3; sub(/\(.*/, "", expr) # the name without its arguments
      printf "| %s | %s | `%s` | %s | %s | %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.2f | %.2g%% |\n", $1, $2, expr, $4, $5, $8,
        $9 * pct, $10 * pct, ($12 + $13 + $14) * pct, $15 * pct, $16 / ($5 * $7), 100 * $6 / (2 * $8 * $4)
    }' "${COUNTS}"
  echo
  echo "</details>"
  echo
  if [ -n "${PREV}" ] && [ -s "${PREV}" ]; then
    if [ "$(head -1 "${PREV}")" != "$(head -1 "${COUNTS}")" ]; then
      echo "**Counter changes**: the counts depend on what changed ($(head -1 "${PREV}" | sed 's/^# //') -> ${META:-unknown}), hence differences are expected and not listed."
    else
      CHANGES=$(awk -F'\t' '
        FNR <= 2 { next }
        NR == FNR { prev[$1 FS $2] = $0; next }
        ($1 FS $2) in prev {
          split(prev[$1 FS $2], p, FS)
          if (p[8] != $8 || p[3] != $3) {
            line = sprintf("- %s %s: %s -> %s instructions (%+.1f%%)", $1, $2, p[8], $8, 0 < p[8] ? 100 * ($8 - p[8]) / p[8] : 0)
            if (p[3] != $3) { a = p[3]; b = $3; sub(/\(.*/, "", a); sub(/\(.*/, "", b); line = line sprintf(", `%s` -> `%s`", a, b) }
            print line
          }
        }' "${PREV}" "${COUNTS}")
      if [ -n "${CHANGES}" ]; then echo "**Counter changes** since the previous passing run:"; echo "${CHANGES}"
      else echo "**Counter changes** since the previous passing run: none."; fi
    fi
    echo
  fi
  # the kernels in terms of the named mixes (scripts/tool_score.py), and the mixes against each other
  if [ -s "${MIXES}" ] && python3 -c "import numpy, scipy" 2> /dev/null; then
    echo "<details><summary>Mixes: each kernel in terms of the named mixes (counted on the same target)</summary>"
    echo
    python3 "${ROOT}/scripts/tool_score.py" project --counts "${MIXES}" --markdown || echo "(tool_score.py failed)"
    echo
    python3 "${ROOT}/scripts/tool_score.py" validate --counts "${MIXES}" --markdown || echo "(tool_score.py failed)"
    echo
    echo "</details>"
    echo
  fi
  echo "<details><summary>Every run</summary>"
  echo
  cat "${RESULTS}"
  echo
  echo "</details>"
  echo
  cat << 'EOF'
**Not covered by the simulator** (its docs/status.md and docs/gap-register.md), hence not shown by an ok or by a score:
- no timing: a native atomic and a compare-and-swap loop look alike, and so do tuned and default parameters
- memory-scope bits have no effect: a scope that is too narrow, or a missing fence, cannot fail here
- atomics serialize on a lock, memory is shared a word at a time, and the race detector is NVIDIA-only: an ok for xchg is no proof against races
- no contention in the score: it is counted one work-group at a time
- an unverified profile's properties (device id, PCI data) come from public documents, not from a card
- out-of-bounds accesses to global memory are not known to be reported
EOF
} > "${SUMMARY}"

exit ${STATUS}
