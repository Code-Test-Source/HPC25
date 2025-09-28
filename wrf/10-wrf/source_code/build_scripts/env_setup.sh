#!/bin/bash

# 环境设置脚本
export WORK_DIR=$(pwd)
export INSTALL_DIR="$WORK_DIR/local"

# 设置编译器
export CC=mpicc
export CXX=mpicxx
export FC=mpif90
export F77=mpif77

# 优化标志
export CFLAGS="-O3 -march=native -fPIC"
export FFLAGS="-O3 -march=native -fPIC"
export FCFLAGS="-O3 -march=native -fPIC"
export CXXFLAGS="-O3 -march=native -fPIC"

# 库路径
export CPPFLAGS="-I$INSTALL_DIR/include"
export LDFLAGS="-L$INSTALL_DIR/lib"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$LD_LIBRARY_PATH"
export PATH="$INSTALL_DIR/bin:$PATH"

# 并行编译
export MAKEFLAGS="-j$(nproc)"

echo "Environment variables set for WRF compilation"