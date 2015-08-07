#!/bin/bash

#TRY="echo"

FILE="benchmark.txt"
SIZE=250
BSIZE=16
STRIDE=1

if [ "" != "$1" ] ; then
  SIZE=$1
  shift
fi
if [ "" != "$1" ] ; then
  BSIZE=$1
  shift
fi
if [ "" != "$1" ] ; then
  STRIDE=$1
  shift
fi

cat /dev/null > ${FILE}

BATCH=${STRIDE}
while [[ ${BATCH} -le ${BSIZE} ]] ; do
  env CHECK=1 ${TRY} \
  ./multi-dgemm.sh ${SIZE} ${BATCH} $* >> ${FILE}
  BATCH=$((BATCH + STRIDE))
done
