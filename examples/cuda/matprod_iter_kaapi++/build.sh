#!/usr/bin/env sh

SCRATCH=$HOME
#SCRATCH=/scratch/jvlima
XKAAPIDIR=$SCRATCH/install/xkaapi/default
CUDADIR=$CUDA_HOME
ATLAS=$SCRATCH/install/atlas3.9.47

#CUBLAS_CFLAGS="-DCONFIG_USE_CUBLAS=1 -maxrregcount 32"
CUBLAS_CFLAGS="-DCONFIG_USE_CUBLAS=1"
#CUBLAS_CFLAGS="-DCONFIG_USE_FLOAT=1"
CUBLAS_LDFLAGS="-lcublas"

CBLAS_LDFLAGS="-L$ATLAS/lib -llapack -lcblas -latlas
	-lf77blas -L$SCRATCH/install/lapacke-3.3.0/lib -llapacke
-llapack -lf77blas -lgfortran"
CBLAS_CPPFLAGS="-I$ATLAS/include
-I$SCRATCH/install/lapacke-3.3.0/include
-DCONFIG_USE_FLOAT=1
"

#MAGMA_CFLAGS="-I${SCRATCH}/install/magma_1.0.0/include -DCONFIG_USE_MAGMA=1"
#MAGMA_LDFLAGS="-L${SCRATCH}/install/magma_1.0.0/lib -lmagma -lmagmablas -llapack -lf77blas -lgfortran"

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
    $CUBLAS_LDFLAGS \
    $CBLAS_LDFLAGS \
    $MAGMA_LDFLAGS
