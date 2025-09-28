#!/bin/bash

# 检查依赖脚本
echo "Checking required dependencies..."

# 检查编译器
which gcc > /dev/null && echo "✓ gcc found" || echo "✗ gcc not found"
which gfortran > /dev/null && echo "✓ gfortran found" || echo "✗ gfortran not found"
which mpicc > /dev/null && echo "✓ mpicc found" || echo "✗ mpicc not found"

# 检查必要的工具
which make > /dev/null && echo "✓ make found" || echo "✗ make not found"
which tar > /dev/null && echo "✓ tar found" || echo "✗ tar not found"

echo "Dependency check completed"