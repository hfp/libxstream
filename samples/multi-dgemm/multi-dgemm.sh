#!/bin/bash

HERE=$(cd $(dirname $0); pwd -P)

export OFFLOAD_INIT=on_start
export MIC_USE_2MB_BUFFERS=2m
export MIC_ENV_PREFIX=MIC
export MIC_KMP_AFFINITY=balanced,granularity=fine

if [[ "" != "$(ldd ${HERE}/${NAME} | grep libiomp5\.so)" ]] ; then
  export KMP_AFFINITY=scatter,granularity=fine,1
else
  export OMP_PROC_BIND=TRUE
fi

if [[ "-test" == "$1" ]] ; then
  export CHECK=1
  TESTS=( \
    "4 1 1" \
    "10 2 2" \
    "14 2 2" \
    "20 1 2" \
    "40 2 2" \
    "40 1 2" \
    "60 2 4" \
    "128 2 2" \
  )
else
  TESTS=( "$*" )
fi

for TEST in "${TESTS[@]}" ; do
  ${HERE}/multi-dgemm ${TEST}
done

