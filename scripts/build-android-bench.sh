#!/bin/bash
# 训练机侧：NDK 交叉编译 ncnn-android(arm64-v8a, Vulkan) + pose_bench
set -e
NDK=/home/ubuntu/android-ndk/android-ndk-r27c
SRC=/home/ubuntu/ncnn-src
BUILD=$SRC/build-android-arm64
OUT=/home/ubuntu/isef/deploy/ncnn/pose-round3/bench-android
TC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
mkdir -p "$OUT"

echo "== cmake configure =="
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 \
  -DNCNN_VULKAN=ON -DNCNN_OPENMP=OFF \
  -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release

echo "== build libncnn.a =="
cmake --build "$BUILD" -j 32

echo "== link pose_bench =="
"$TC/aarch64-linux-android24-clang++" -O2 -static-libstdc++ \
  -o "$OUT/pose_bench" /home/ubuntu/isef/scripts/pose_bench.cpp \
  -I "$SRC/src" -I "$BUILD/src" "$BUILD/src/libncnn.a" \
  "$BUILD/glslang/glslang/libglslang.a" \
  "$BUILD/glslang/glslang/libMachineIndependent.a" \
  "$BUILD/glslang/glslang/libGenericCodeGen.a" \
  "$BUILD/glslang/glslang/OSDependent/Unix/libOSDependent.a" \
  "$BUILD/glslang/SPIRV/libSPIRV.a" \
  "$BUILD/glslang/glslang/libglslang-default-resource-limits.a" -landroid -llog
"$TC/llvm-strip" "$OUT/pose_bench"
"$TC/llvm-readelf" -h "$OUT/pose_bench" | grep -E 'Class|Machine|Type'
echo "== done =="
ls -la "$OUT"
