#!/bin/bash

CXX=$(which icpc 2> /dev/null)

ICCOPT="-O2 -xHost -ansi-alias -DNDEBUG"
GCCOPT="-O2 -march=native -DNDEBUG"
LIBXSTREAM_ROOT="../.."

if [ "" = "$CXX" ] ; then
  OPT=$GCCOPT
  CXX="g++"
else
  OPT=$ICCOPT
fi

if [ "-g" = "$1" ] ; then
  OPT="-O0 -g"
  shift
fi

$CXX -std=c++0x $OPT $* -lpthread \
  -I$LIBXSTREAM_ROOT/include -I$LIBXSTREAM_ROOT/src -DLIBXSTREAM_EXPORTED \
  $LIBXSTREAM_ROOT/src/*.cpp test.cpp \
  -o test
