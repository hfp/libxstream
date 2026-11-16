#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Builds PoCL with two drivers and installs it into PREFIX:
#
#   cpu    the host CPU, with the kernel library built for every x86 variant
#          (KERNELLIB_HOST_CPU_VARIANTS=distro), hence a build made on one
#          machine runs on another (a cached build in CI, for instance)
#   cuda   NVIDIA GPUs (OpenCL C to PTX by LLVM), which needs a libcuda only
#          at run time, e.g., the simulated GPUs of PantheonSim
#
# The two parts of NVIDIA's CUDA toolkit PoCL needs (cuda.h, and libdevice for
# the math builtins) come from NVIDIA's redistributable archives and are kept
# in PREFIX/cuda, since PoCL looks for libdevice there at run time. Everything
# is downloaded by URL and checked against its SHA-256.
#
# Use: OCL_ICD_VENDORS=PREFIX/vendors (the ICD file this script writes), and
# POCL_DEVICES=cpu or POCL_DEVICES=cuda to select a driver. For a GPU beyond
# SM 7.5, POCL_CUDA_GPU_ARCH (e.g., sm_90) lifts PoCL's default cap at sm_75.
#
# LLVM_CONFIG selects LLVM (default: the newest llvm-config-NN found, NN from
# 23 down to 18, which is what PoCL accepts); CMake, a C/C++ compiler, and
# the development files of LLVM, Clang, and hwloc are prerequisites. JOBS sets
# the build's parallelism (default: all cores).
###############################################################################

POCL_VERSION=7.2
POCL_SHA256=7ddc01a7afcd49d4ec7fa9bf94df20852db44fcda629a9763086a8c541e5da1e
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/redist
CUDA_ARCHIVES="
f74689258a60fd9c5bdfa7679458527a55e22442691ba678dcfaeffbf4391ef9 cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.6.77-archive.tar.xz
840deff234d9bef20d6856439c49881cb4f29423b214f9ecd2fa59b7ac323817 cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.6.85-archive.tar.xz
"

PREFIX=$1
if [ -z "${PREFIX}" ]; then
  echo "Usage: $0 PREFIX"
  exit 1
fi
mkdir -p "${PREFIX}" || exit 1
PREFIX=$(cd "${PREFIX}" && pwd -P)
WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT

# fetch URL FILE SHA256
fetch() {
  curl -fsSL --retry 5 --retry-all-errors -o "$2" "$1" && echo "$3  $2" | sha256sum -c -
}

if [ -z "${LLVM_CONFIG}" ]; then
  for VERSION in 23 22 21 20 19 18; do
    if command -v "llvm-config-${VERSION}" > /dev/null; then
      LLVM_CONFIG=$(command -v "llvm-config-${VERSION}")
      break
    fi
  done
fi
if [ -z "${LLVM_CONFIG}" ] || [ ! -x "${LLVM_CONFIG}" ]; then
  echo "ERROR: no LLVM between 18 and 23 found (LLVM_CONFIG)"
  exit 1
fi

# cuda.h and libdevice
mkdir -p "${PREFIX}/cuda" || exit 1
while read -r SHA256 ARCHIVE; do
  if [ -z "${ARCHIVE}" ]; then continue; fi
  fetch "${CUDA_URL}/${ARCHIVE}" "${WORK}/${ARCHIVE##*/}" "${SHA256}" || exit 1
  tar -xJf "${WORK}/${ARCHIVE##*/}" -C "${PREFIX}/cuda" --strip-components=1 || exit 1
done <<< "${CUDA_ARCHIVES}"

fetch "https://github.com/pocl/pocl/archive/refs/tags/v${POCL_VERSION}.tar.gz" \
  "${WORK}/pocl.tar.gz" "${POCL_SHA256}" || exit 1
tar -xzf "${WORK}/pocl.tar.gz" -C "${WORK}" || exit 1
SRC=${WORK}/pocl-${POCL_VERSION}

# Clang 18 and 19 accept the .sync intrinsics of the SM 7.0 library only with
# a PTX version as a target feature, which PoCL does not pass. The flag cannot
# come in through KERNEL_CL_FLAGS, since the CPU library is built with it too.
CUDALIB=${SRC}/lib/kernel/cuda/CMakeLists.txt
if ! awk '/^make_kernel_bc\(KERNEL_BC_SM70_PTX64 / {
      print "list(APPEND CLANG_FLAGS \"-Xclang\" \"-target-feature\" \"-Xclang\" \"+ptx64\")"; ++n }
    { print } END { exit (1 == n ? 0 : 1) }' "${CUDALIB}" > "${CUDALIB}.new"; then
  echo "ERROR: PoCL ${POCL_VERSION} does not build the SM 7.0 library as expected"
  exit 1
fi
mv "${CUDALIB}.new" "${CUDALIB}" || exit 1

# replace FILE TEXT LINE: the one line containing TEXT becomes LINE (no escapes)
replace() {
  if ! awk -v text="$2" -v line="$3" 'index($0, text) { print line; ++n; next }
      { print } END { exit (1 == n ? 0 : 1) }' "$1" > "$1.new"; then
    echo "ERROR: PoCL ${POCL_VERSION} differs from what this script patches (${2})"
    exit 1
  fi
  mv "$1.new" "$1" || exit 1
}

# PTX beyond 8.0, which SM 10.0 and later require (8.6 to 9.0, the minimum LLVM
# enforces per target): PoCL's table stops at CUDA 12.0. The version stays 8.0
# where an SM allows it, so what older targets get does not change.
CUDADEV=${SRC}/lib/CL/devices/cuda/pocl-cuda.c
replace "${CUDADEV}" "{ 12000, 80 }, { 11000, 70 }, { 10000, 63 }, { 9000, 60 }, { 8000, 50 }," \
  "  { 13000, 90 }, { 12090, 88 }, { 12080, 87 }, { 12070, 86 }, { 12000, 80 }, { 11000, 70 }, { 10000, 63 }, { 9000, 60 }, { 8000, 50 },"
replace "${CUDADEV}" "if (sm_version >= 89)" \
  "  if (sm_version >= 121) min_ptx = 88; else if (sm_version >= 120) min_ptx = 87; else if (sm_version >= 110) min_ptx = 90; else if (sm_version >= 103) min_ptx = 88; else if (sm_version >= 100) min_ptx = 86; else if (sm_version >= 89)"
replace "${CUDADEV}" "unsigned selected_ptx = max_ptx;" \
  "  unsigned selected_ptx = (80 < max_ptx && 80 >= min_ptx) ? 80 : max_ptx;"

# Some distributions ship only the versioned Clang library without its development link.
LIBDIR=$("${LLVM_CONFIG}" --libdir)
CLANG_CPP=""
if [ ! -e "${LIBDIR}/libclang-cpp.so" ]; then
  CLANG_CPP=$(find "${LIBDIR}" -maxdepth 1 -name 'libclang-cpp.so.*' | sort | head -n1)
fi
if command -v ninja > /dev/null; then GENERATOR=Ninja; else GENERATOR="Unix Makefiles"; fi

cmake -S "${SRC}" -B "${WORK}/build" -G "${GENERATOR}" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DWITH_LLVM_CONFIG="${LLVM_CONFIG}" ${CLANG_CPP:+"-DC_LIBFILE_clang-cpp=${CLANG_CPP}"} \
  -DENABLE_HOST_CPU_DEVICES=ON -DKERNELLIB_HOST_CPU_VARIANTS=distro \
  -DENABLE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR="${PREFIX}/cuda" -DCUDAToolkit_ROOT="${PREFIX}/cuda" \
  -DENABLE_ICD=ON -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF || exit 1
cmake --build "${WORK}/build" --parallel "${JOBS:-$(nproc)}" || exit 1
cmake --install "${WORK}/build" > /dev/null || exit 1

mkdir -p "${PREFIX}/vendors" || exit 1
echo "${PREFIX}/lib/libpocl.so" > "${PREFIX}/vendors/pocl.icd"
echo "PoCL ${POCL_VERSION} (LLVM $("${LLVM_CONFIG}" --version)) in ${PREFIX}: OCL_ICD_VENDORS=${PREFIX}/vendors"
