#!/bin/sh
# Compile ozaki1 TinyTC kernel for f64 and f32.
#
# Usage: ./ozaki1.sh [tinytc-compiler]
#   Default compiler: ~/tinytc/bin/tinytc (or $TINYTC env var).
#
# BM and BN are read from ozaki1.tinytc and encoded in the filename,
# e.g. ozaki1_f64_256x128.clx.  The Makefile parses them back into
# -DOZAKI_TINYTC_BM / -DOZAKI_TINYTC_BN so host dispatch matches.

TINYTC="${1:-${TINYTC:-$HOME/tinytc/bin/tinytc}}"
DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${DIR}/ozaki1.tinytc"

if [ ! -f "$SRC" ]; then
  echo "ERROR: $SRC not found" >&2; exit 1
fi
if [ ! -x "$TINYTC" ]; then
  echo "ERROR: TinyTC compiler not found: $TINYTC" >&2; exit 1
fi

BM=$(grep '^\$BM' "$SRC" | sed 's/.*= *//')
BN=$(grep '^\$BN' "$SRC" | sed 's/.*= *//')
if [ -z "$BM" ] || [ -z "$BN" ]; then
  echo "ERROR: cannot extract \$BM/\$BN from $SRC" >&2; exit 1
fi

for ty in f64 f32; do
  OUT="${DIR}/ozaki1_${ty}_${BM}x${BN}.clx"
  printf '$ty = %s\n' "$ty" | cat - "$SRC" | \
    "$TINYTC" -d pvc -F large-register-file -o "$OUT" && \
    echo "OK: $OUT (BM=${BM}, BN=${BN})" || { echo "FAIL: $OUT" >&2; exit 1; }
done
