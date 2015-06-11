#!/bin/bash

export KMP_AFFINITY=scatter
export OFFLOAD_INIT=on_start
export MIC_USE_2MB_BUFFERS=2m
export MIC_ENV_PREFIX=MIC
export MIC_KMP_AFFINITY=balanced,granularity=fine

HERE=$(cd $(dirname $0); pwd -P)

if [[ "-test" == "$1" ]] ; then
  TESTS=( \
    "16 1 2 4" \
    "19 1 4 16" \
    "45 1 4 4" \
    "25 1 13 3" \
    "16 1 2 17" \
    "45 1 4 16" \
    "25 1 13 12" \
  )
else
  TESTS=( "$*" )
fi

for TEST in "${TESTS[@]}" ; do
  ${HERE}/entropy ${TEST}
done

