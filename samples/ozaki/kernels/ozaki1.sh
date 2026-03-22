#!/bin/sh
# Compile ozaki1 TinyTC kernels.
#
# Usage:
#   ./ozaki1.sh [tinytc-compiler]
#       Compile general f64/f32 kernels from ozaki1.tinytc.
#
#   ./ozaki1.sh -p [header-out] [tinytc-compiler]
#       Compile production .clx matrix + generate C header.
#       header-out defaults to ../ozaki_tinytc.h.
#
#   ./ozaki1.sh -h [header-out]
#       Generate C header from existing production .clx files only
#       (no compilation, no tinytc needed).  Used by Makefile.
#
# Default compiler: ~/tinytc/bin/tinytc (or $TINYTC env var).
#
# General mode: BM/BN are read from ozaki1.tinytc and encoded in the
# filename (e.g. ozaki1_f64_256x128.clx).  The Makefile parses them
# back into -DOZAKI_TINYTC_BM / -DOZAKI_TINYTC_BN.
#
# Production mode (-p): compiles specialized kernels for every valid
# (type, nslices, trim, scheme) combination and emits a header with
# INCBIN declarations and a static lookup table so all kernels are
# embedded in the binary at compile time.

DIR="$(cd "$(dirname "$0")" && pwd)"
MODE=general
HEADER=""

if [ "$1" = "-p" ]; then
  MODE=prod; shift
  HEADER="${1:-${DIR}/../ozaki_tinytc.h}"; shift
elif [ "$1" = "-h" ]; then
  MODE=header; shift
  HEADER="${1:-${DIR}/../ozaki_tinytc.h}"; shift
fi

if [ "$MODE" != "header" ]; then
  TINYTC="${1:-${TINYTC:-$HOME/tinytc/bin/tinytc}}"
  if ! command -v "$TINYTC" >/dev/null 2>&1; then
    echo "ERROR: TinyTC compiler not found: $TINYTC" >&2; exit 1
  fi
fi

# --- General mode: compile ozaki1.tinytc for f64 and f32 --------------------
if [ "$MODE" = "general" ]; then
  SRC="${DIR}/ozaki1.tinytc"
  if [ ! -f "$SRC" ]; then
    echo "ERROR: $SRC not found" >&2; exit 1
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
      echo "OK: $OUT (BM=${BM}, BN=${BN})" || \
      { echo "FAIL: $OUT" >&2; exit 1; }
  done
  exit 0
fi

# --- Production mode (-p): compile all valid combos -------------------------
if [ "$MODE" = "prod" ]; then
BM=256
BN=128
dp_nslices="7 8 9"
sp_nslices="3 4 5"
trims="0 3 7"
schemes="tri sq"
ok=0; skip=0; fail=0

for ty in f64 f32; do
  if [ "$ty" = "f64" ]; then ns="$dp_nslices"; else ns="$sp_nslices"; fi
  for n in $ns; do
    for t in $trims; do
      cutoff=$((2 * (n - 1) - t))
      if [ "$cutoff" -lt 0 ]; then
        skip=$((skip + 1)); continue
      fi
      for scheme in $schemes; do
        src="${DIR}/ozaki1_prod_${scheme}.tinytc"
        out="${DIR}/ozaki1_${ty}_${BM}x${BN}_n${n}t${t}_${scheme}.clx"
        if printf '$ty = %s\n$nslices = %d\n$cutoff = %d\n' \
             "$ty" "$n" "$cutoff" \
           | cat - "$src" \
           | "$TINYTC" -d pvc -F large-register-file -o "$out" 2>/dev/null
        then
          ok=$((ok + 1))
        else
          echo "FAIL: $(basename "$out")  (n=$n t=$t cutoff=$cutoff)" >&2
          fail=$((fail + 1))
        fi
      done
    done
  done
done
echo "$ok OK, $skip skipped, $fail failed" >&2
fi

# --- Header generation (used by both -p and -h) -----------------------------
FILES=$(ls "${DIR}"/ozaki1_*_n*t*_tri.clx "${DIR}"/ozaki1_*_n*t*_sq.clx 2>/dev/null | sort)
if [ -z "$FILES" ]; then
  echo "/* ozaki_tinytc.h -- no production .clx files */" >"$HEADER"
  exit 0
fi

{
  cat <<'EOF'
/* ozaki_tinytc.h -- auto-generated, do not edit */
#if defined(LIBXS_INCBIN) && defined(OZAKI_TINYTC_EMBED)
EOF

  for f in $FILES; do
    base=$(basename "$f" .clx)
    ty=$(echo "$base" | sed 's/ozaki1_\(f[0-9]*\)_.*/\1/')
    tag=$(echo "$base" | sed 's/ozaki1_f[0-9]*_[0-9]*x[0-9]*_//')
    echo "LIBXS_INCBIN(ozaki_prod_${ty}_${tag}, \"kernels/${base}.clx\", 16);"
  done

  cat <<'EOF'

static const struct ozaki_tinytc_prod_entry {
  int use_double;
  int ndecomp;
  int oztrim;
  int sq;
  const char *data;
  const char *data_end;
} ozaki_tinytc_prod[] = {
EOF

  for f in $FILES; do
    base=$(basename "$f" .clx)
    ty=$(echo "$base" | sed 's/ozaki1_\(f[0-9]*\)_.*/\1/')
    tag=$(echo "$base" | sed 's/ozaki1_f[0-9]*_[0-9]*x[0-9]*_//')
    n=$(echo "$tag" | sed 's/n\([0-9]*\)t.*/\1/')
    t=$(echo "$tag" | sed 's/n[0-9]*t\([0-9]*\)_.*/\1/')
    scheme=$(echo "$tag" | sed 's/n[0-9]*t[0-9]*_//')
    sym="ozaki_prod_${ty}_${tag}"
    if [ "$ty" = "f64" ]; then ud=1; else ud=0; fi
    if [ "$scheme" = "sq" ]; then sq=1; else sq=0; fi
    echo "  { $ud, $n, $t, $sq, $sym, ${sym}_end },"
  done

  cat <<'EOF'
  { 0, 0, 0, 0, NULL, NULL }
};

#endif
EOF
} >"$HEADER"
echo "Generated $HEADER" >&2
