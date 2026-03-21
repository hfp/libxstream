#!/bin/sh
# Compile ozaki1 TinyTC kernel for f64 and f32.
#
# Usage: ./ozaki1.sh [tinytc-compiler]
#   Default compiler: ~/tinytc/bin/tinytc (or $TINYTC env var).
#
# Produces:
#   ozaki1_f64.clx   (double precision)
#   ozaki1_f32.clx   (single precision)

TINYTC="${1:-${TINYTC:-$HOME/tinytc/bin/tinytc}}"
DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${DIR}/ozaki1.tinytc"

if [ ! -f "$SRC" ]; then
  echo "ERROR: $SRC not found" >&2; exit 1
fi
if [ ! -x "$TINYTC" ]; then
  echo "ERROR: TinyTC compiler not found: $TINYTC" >&2; exit 1
fi

for ty in f64 f32; do
  OUT="${DIR}/ozaki1_${ty}.clx"
  printf '$ty = %s\n' "$ty" | cat - "$SRC" | \
    "$TINYTC" -d pvc -F large-register-file -o "$OUT" && \
    echo "OK: $OUT" || { echo "FAIL: $OUT" >&2; exit 1; }
done
