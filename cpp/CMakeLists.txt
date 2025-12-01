cmake_minimum_required(VERSION 3.10)
project(BVC_Core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Core Library
file(GLOB CORE_SOURCES "core/*.cpp")
add_library(bvc_core ${CORE_SOURCES})

# Optimization flags
if(MSVC)
    target_compile_options(bvc_core PRIVATE /O2 /Ob2 /Oi /Ot /Oy /GL /GF /Gy /arch:AVX2 /fp:fast /Qpar /DNDEBUG /openmp /favor:INTEL64 /MP)
    add_link_options(/LTCG)
else()
    target_compile_options(bvc_core PRIVATE -O3 -ffast-math -mavx2 -mfma -funroll-loops -flto -fopenmp)
    target_link_libraries(bvc_core PUBLIC gomp) 
endif()

# CLI Executable
add_executable(bvc cli/main.cpp)
target_link_libraries(bvc bvc_core)

# Test Executable
add_executable(bvc_test tests/test_main.cpp)
target_link_libraries(bvc_test bvc_core)