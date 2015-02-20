#!/bin/bash

LIBXSTREAM_ROOT="../.."

ICCOPT="-O2 -xHost -ansi-alias -mkl"
ICCLNK="-mkl"

GCCOPT="-O2 -march=native"
GCCLNK="-llapack -lblas"

if [[ "" = "$CXX" ]] ; then
  CXX=$(which icpc 2> /dev/null)
  if [[ "" != "$CXX" ]] ; then
    OPT=$ICCOPT
    LNK=$ICCLNK
  else
    CXX="g++"
  fi
fi

if [[ "" = "$OPT" || "" = "$LNK" ]] ; then
  OPT=$GCCOPT
  LNK=$GCCLNK
fi

if [ "-g" = "$1" ] ; then
  OPT+=" -O0 -g"
  shift
else
  OPT+=" -DNDEBUG"
fi

$CXX -Wall -std=c++0x $OPT $* -lpthread \
  -I$LIBXSTREAM_ROOT/include -I$LIBXSTREAM_ROOT/src -DLIBXSTREAM_EXPORTED \
  $LIBXSTREAM_ROOT/src/*.cpp multi-dgemm-type.cpp multi-dgemm.cpp \
  $LNK -o multi-dgemm
