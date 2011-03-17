#!/usr/bin/env sh

XKAAPIDIR=$HOME/install/xkaapi_gpu
CUDADIR=/usr/local/stow/cuda-4.0/cuda

#CUBLAS_CFLAGS="-I$CUDADIR/include -DCONFIG_USE_CUBLAS=1"
#CUBLAS_LFLAGS="-L$CUDADIR/lib64 -lcublas"
CUBLAS_CFLAGS=
CUBLAS_LFLAGS=

VOLKOV_CFLAGS="-DCONFIG_USE_VOLKOV=1 -maxrregcount 32"
#VOLKOV_CFLAGS=

$CUDADIR/bin/nvcc -w \
    -I$XKAAPIDIR/include \
    -I$CUDADIR/include \
    $CUBLAS_CFLAGS \
    $VOLKOV_CFLAGS \
    -DKAAPI_NDEBUG=1 \
    -DKAAPI_DEBUG=0 \
    matprod_iter_kaapi++.cu \
    volkov_sgemm.cu \
    -L$XKAAPIDIR/lib -lkaapi -lkaapi++ \
    -L$CUDADIR/lib64 -lcuda \
    $CUBLAS_LFLAGS
