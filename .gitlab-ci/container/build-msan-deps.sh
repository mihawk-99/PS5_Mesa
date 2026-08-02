#!/usr/bin/env bash

set -ex

section_start msan-deps "Building MSAN libc++"

if [ "${LLVM_VERSION:?llvm version not set}" -ge 18 ]; then
  VER="${LLVM_VERSION}.1.0"
else
  VER="${LLVM_VERSION}.0.0"
fi

# Download the LLVM project source code
curl -L --retry 4 -f --retry-all-errors --retry-delay 60 \
    -O "https://github.com/llvm/llvm-project/releases/download/llvmorg-${VER}/llvm-project-${VER}.src.tar.xz"

# Extract it and remove the archive
tar -xf "llvm-project-${VER}.src.tar.xz" && rm "llvm-project-${VER}.src.tar.xz"

mkdir -p "llvm-project-${VER}.src/build"
pushd "llvm-project-${VER}.src/build"

# Configure libc++ and libc++abi with MSAN enabled
cmake ../runtimes -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi" \
    -DLLVM_USE_SANITIZER=MemoryWithOrigins \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_INSTALL_PREFIX=/msan

ninja install
popd

# Clean up to save space in the container
du -sh "llvm-project-${VER}.src"
rm -rf "llvm-project-${VER}.src"

section_end msan-deps
