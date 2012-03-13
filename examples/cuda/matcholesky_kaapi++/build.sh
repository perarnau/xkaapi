#!/bin/bash

SCRATCH=/scratch/jvlima
XKAAPIDIR=$SCRATCH/install/xkaapi/default
CUDADIR=$CUDA_HOME

CUDA_CFLAGS="-DCONFIG_USE_CUDA=1 -I$CUDADIR/include"
CUDA_LDFLAGS="-L$CUDADIR/lib64 -lcuda"

#CUBLAS_CFLAGS="-DCONFIG_USE_CUBLAS=1"
CUBLAS_LDFLAGS="-lcublas"

CBLAS_LDFLAGS="-L$SCRATCH/install/atlas3.9.69/lib -llapack -lcblas -latlas
-lf77blas
-L$SCRATCH/install/lapacke/lib -llapacke
-llapack -lf77blas -lgfortran
"
CBLAS_CPPFLAGS="-I$SCRATCH/install/atlas3.9.69/include
-I$SCRATCH/install/lapacke/include
-DCONFIG_USE_FLOAT=1
"

MAGMA_CFLAGS="-I${SCRATCH}/install/magma_1.1.0/include -DCONFIG_USE_MAGMA=1"
MAGMA_LDFLAGS="-L${SCRATCH}/install/magma_1.1.0/lib -lmagma -lmagmablas
-llapack -lf77blas -lgfortran"

#$CUDADIR/bin/nvcc -g -w \
g++ -g -Wall \
    -I$XKAAPIDIR/include \
    $CUDA_CFLAGS \
    $CBLAS_CPPFLAGS \
    $CUBLAS_CFLAGS \
    $MAGMA_CFLAGS \
    -DKAAPI_NDEBUG=1 \
    -DKAAPI_DEBUG=0 \
	-o matcholesky_kaapi++ \
	matcholesky_kaapi++.cpp \
    -L$XKAAPIDIR/lib -lkaapi -lkaapi++ \
    $CUDA_LDFLAGS \
    $CBLAS_LDFLAGS \
    $CUBLAS_LDFLAGS \
    $MAGMA_LDFLAGS 
