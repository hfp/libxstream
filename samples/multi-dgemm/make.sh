#!/bin/bash

CXX=$(which icpc 2> /dev/null)

ICCOPT="-O2 -xHost -ansi-alias -DNDEBUG -mkl"
GCCOPT="-O2 -march=native -DNDEBUG -llapack -lblas"
ROOT="../.."

if [ "" = "$CXX" ] ; then
  OPT=$GCCOPT
  CXX="g++"
else
  OPT=$ICCOPT
fi

if [ "-g" = "$1" ] ; then
  OPT+=" -O0 -g"
  shift
fi

$CXX -std=c++0x $OPT $* -lpthread \
  -D__ACC -D__ACC_MIC \
  -I$ROOT/include $ROOT/src/*.cpp \
  multi-dgemm-type.cpp \
  multi-dgemm.cpp \
  -o multi-dgemm
