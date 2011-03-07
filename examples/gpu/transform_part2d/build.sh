#!/usr/bin/env sh

XKAAPIDIR=$HOME/install/xkaapi_gpu
CUDADIR=/usr/local/stow/cuda-4.0/cuda

$CUDADIR/bin/nvcc -w \
    -I$XKAAPIDIR/include \
    -I$CUDADIR/include \
    -DKAAPI_NDEBUG=1 \
    -DKAAPI_DEBUG=0 \
    transform.cu \
    -L$XKAAPIDIR/lib -lkaapi++ \
    -L$CUDADIR/lib64 -lcuda
