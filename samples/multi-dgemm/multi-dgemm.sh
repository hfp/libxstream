#!/bin/bash

export KMP_AFFINITY=scatter
export OFFLOAD_INIT=on_start
export MIC_USE_2MB_BUFFERS=2m
export MIC_ENV_PREFIX=MIC
export MIC_KMP_AFFINITY=balanced,granularity=fine

HERE=$(cd $(dirname $0); pwd -P)

if [[ "-test" == "$1" ]] ; then
  export CHECK=1
  TESTS=( \
    "4 1 1 4" \
    "10 2 2 4" \
    "14 2 2 4" \
    "20 1 2 4" \
    "30 2 2 4" \
    "40 2 2 4" \
    "40 1 2 32" \
    "60 2 4 4" \
    "128 2 2 4" \
  )
else
  TESTS=( "$*" )
fi

for TEST in "${TESTS[@]}" ; do
  ${HERE}/multi-dgemm ${TEST}
done

