#!/usr/bin/env sh

XKAAPIDIR=$HOME/install/xkaapi/default
CUDADIR=$CUDA_HOME/cuda

#CUBLAS_CFLAGS="-DCONFIG_USE_CUBLAS=1 -maxrregcount 32"
#CUBLAS_CFLAGS="-DCONFIG_USE_CUBLAS=1 -DCONFIG_USE_FLOAT=1"
#CUBLAS_CFLAGS="-DCONFIG_USE_FLOAT=1"
CUBLAS_LDFLAGS="-lcublas"

CBLAS_LDFLAGS="-L$HOME/install/atlas3.9.47/lib -llapack -lcblas -latlas
	-lf77blas -L$HOME/install/lapacke-3.3.0/lib -llapacke"
CBLAS_CPPFLAGS="-I$HOME/install/atlas3.9.47/include
-I$HOME/install/lapacke-3.3.0/include
-DCONFIG_USE_FLOAT=1
"

MAGMA_CFLAGS="-I${HOME}/install/magma_1.0.0/include -DCONFIG_USE_MAGMA=1"
MAGMA_LDFLAGS="-L${HOME}/install/magma_1.0.0/lib -lmagma -lmagmablas -llapack -lf77blas -lgfortran"

# use this flag for atomic instruction 
# does not work on gtx280
# -arch=compute_20

#$CUDADIR/bin/nvcc -g -w \
g++ -g -Wall \
    -I$XKAAPIDIR/include \
    -I$CUDADIR/include \
    $CBLAS_CPPFLAGS \
    $CUBLAS_CFLAGS \
    $MAGMA_CFLAGS \
    -DKAAPI_NDEBUG=1 \
    -DKAAPI_DEBUG=0 \
    -o matprod_iter_kaapi++ \
    matprod_iter_kaapi++.cpp \
    -L$XKAAPIDIR/lib -lkaapi -lkaapi++ \
    -L$CUDADIR/lib64 -lcuda \
    $CBLAS_LDFLAGS \
    $CUBLAS_LDFLAGS \
    $MAGMA_LDFLAGS
