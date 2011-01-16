#!/usr/bin/env sh

XKAAPIDIR=$HOME/install/xkaapi_gpu
CUDADIR=/usr/local/stow/cuda-3.2/cuda

$CUDADIR/bin/nvcc --ptx transform.cu

gcc -Wall -std=c99 \
    -I$XKAAPIDIR/include \
    -I$CUDADIR/include \
    -DKAAPI_NDEBUG=1 \
    -DKAAPI_DEBUG=0 \
    transform.c \
    -L$XKAAPIDIR/lib -lkaapi \
    -L$CUDADIR/lib64 -lcuda
