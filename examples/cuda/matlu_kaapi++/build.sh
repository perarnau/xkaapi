#!/bin/bash

XKAAPIDIR=$HOME/install/xkaapi/default

CXX=g++
NVCC=nvcc

#CFLAGS="-DKAAPI_DEBUG=0 -DKAAPI_NDEBUG=1 -DCONFIG_USE_DOUBLE=1 -I$XKAAPIDIR/include"
CFLAGS="-DKAAPI_DEBUG=0 -DKAAPI_NDEBUG=1 -DCONFIG_USE_FLOAT=1 -I$XKAAPIDIR/include"
CFLAGS="$CFLAGS -DCONFIG_VERBOSE=1 -DCONFIG_DEBUG=1"
LDFLAGS="-L$XKAAPIDIR/lib -lkaapi -lkaapi++"
#LDFLAGS="-L$XKAAPIDIR/lib -lkaapi -lkaapi++ -lgfortran"

PLASMA_CFLAGS="-DCONFIG_USE_PLASMA=1 $PLASMA_CFLAGS"
PLASMA_LDFLAGS="$PLASMA_LDFLAGS -lplasma"

#MAGMA_CFLAGS="-DCONFIG_USE_MAGMA=1 $MAGMA_CFLAGS"

CUBLAS_CFLAGS="-DCONFIG_USE_CUBLAS=1"
CUBLAS_LDFLAGS="-lcublas"
CUDA_CFLAGS="-DCONFIG_USE_CUDA=1 $CUDA_CFLAGS"

$NVCC -g --machine 64 -arch sm_20 -DGPUSHMEM=200 -DBLOCK_SIZE=64 \
  --compiler-options "-Wall $CFLAGS" \
  -c ../matrix/magmablas/magmablas.cu

$CXX -g -Wall \
    $CFLAGS \
    $MAGMA_CFLAGS \
    $PLASMA_CFLAGS \
    $CUDA_CFLAGS \
    $CBLAS_CFLAGS \
    $CUBLAS_CFLAGS \
    $LAPACK_CLAGS \
    $LAPACKE_CFLAGS \
    -c matlu_kaapi++.cpp 


$CXX -g \
    -o matlu_kaapi++ \
    matlu_kaapi++.o \
    magmablas.o \
    $LDFLAGS \
    $MAGMA_LDFLAGS \
    $PLASMA_LDFLAGS \
    $CUDA_LDFLAGS \
    $CUBLAS_LDFLAGS \
    $LAPACKE_LDFLAGS \
    $LAPACK_LDFLAGS  \
    $CBLAS_LDFLAGS  /usr/local/Cellar/gfortran/4.2.4-5666.3/lib/gcc/i686-apple-darwin11/4.2.1/x86_64/libgfortran.a

