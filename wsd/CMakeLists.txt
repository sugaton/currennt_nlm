PROJECT(nnlmwsd)

CMAKE_MINIMUM_REQUIRED (VERSION 2.8)

FIND_PACKAGE (CUDA)
FIND_PACKAGE (Boost 1.48.0 REQUIRED COMPONENTS program_options system filesystem random thread  REQUIRED)

INCLUDE_DIRECTORIES (${Boost_INCLUDE_DIR})

LINK_DIRECTORIES (${Boost_LIBRARY_DIR})

SET (CMAKE_BUILD_TYPE Release)
SET (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 -std=c++11 -DNEWSOFTMAX -L/home/lr/suga/local/src/cuda/lib64 -I/usr/include -lcudnn ")
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format-security -std=c++11 -g -I/home/lr/suga/local/include -L/home/lr/suga/local/src/cuda/lib64 -lcudnn ")

FILE (GLOB_RECURSE src_lib     ../currennt_lib/*.cpp ../currennt_lib/*.hpp     ../currennt_lib/*.h     ../currennt_lib/*.cu     ../currennt_lib/*.cuh)
FILE (GLOB_RECURSE main wsd.cpp)


CUDA_ADD_EXECUTABLE (${PROJECT_NAME} ${src_lib} ${main})

CUDA_ADD_CUBLAS_TO_TARGET (${PROJECT_NAME})
TARGET_LINK_LIBRARIES (${PROJECT_NAME} netcdf)
TARGET_LINK_LIBRARIES (${PROJECT_NAME} ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES (${PROJECT_NAME} cudnn)


