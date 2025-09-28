project(RopeSimulator)
cmake_minimum_required(VERSION 3.15)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置优化级别和架构特定优化
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native -ffast-math")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math")

# 启用OpenMP并行化
find_package(OpenMP REQUIRED)

include_directories(.)

add_executable(${PROJECT_NAME}
    main.cpp
    acceleration.cpp
    calGF.cpp
    LoadFNS.cpp
    LoadPolyhedron.cpp
    LoadSHPara.cpp
    MatrixOperations.cpp
    OdeEuler.cpp
    SurfacePoints.cpp
    initialize.cpp
)

# 链接OpenMP
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)

# 设置目标属性以启用向量化
target_compile_options(${PROJECT_NAME} PRIVATE 
    -mavx2 -mfma -ftree-vectorize -fopt-info-vec
)