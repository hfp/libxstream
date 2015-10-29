#!/bin/bash

HERE=$(cd $(dirname $0); pwd -P)
NAME=$(basename ${HERE})

if [[ "-mic" != "$1" ]] ; then
  if [[ "$1" == "o"* ]] ; then
    FILE=copyout.dat
  else
    FILE=copyin.dat
  fi
  if [[ "" != "$(ldd ${HERE}/${NAME} | grep libiomp5\.so)" ]] ; then
    env OFFLOAD_INIT=on_start \
      KMP_AFFINITY=scatter,granularity=fine,1 \
      MIC_KMP_AFFINITY=scatter,granularity=fine \
      MIC_ENV_PREFIX=MIC \
    ${HERE}/${NAME} $* | \
    tee ${FILE}
  else
    env \
      OMP_PROC_BIND=TRUE \
    ${HERE}/${NAME} $* | \
    tee ${FILE}
  fi
else
  shift
  if [[ "$1" == "o"* ]] ; then
    FILE=copyout.dat
  else
    FILE=copyin.dat
  fi
  env \
    SINK_LD_LIBRARY_PATH=$MIC_LD_LIBRARY_PATH \
  micnativeloadex \
    ${HERE}/${NAME} -a "$*" \
    -e "KMP_AFFINITY=scatter,granularity=fine" | \
  tee ${FILE}
fi

