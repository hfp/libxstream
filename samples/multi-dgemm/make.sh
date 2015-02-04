#!/bin/bash

ROOT="../.."

CXX=$(which icpc 2> /dev/null)

ICCOPT="-O2 -xHost -ansi-alias -mkl"
ICCLNK="-mkl"

GCCOPT="-O2 -march=native"
GCCLNK="-llapack -lblas"

if [ "" = "$CXX" ] ; then
  OPT=$GCCOPT
  LNK=$GCCLNK
  CXX="g++"
else
  OPT=$ICCOPT
  LNK=$ICCLNK
fi

if [ "-g" = "$1" ] ; then
  OPT+=" -O0 -g"
  shift
else
  OPT+=" -DNDEBUG"
fi

$CXX -std=c++0x $OPT $* -lpthread \
  -D__ACC -D__ACC_MIC -I$ROOT/include \
  $ROOT/src/*.cpp multi-dgemm-type.cpp multi-dgemm.cpp \
  $LNK -o multi-dgemm
