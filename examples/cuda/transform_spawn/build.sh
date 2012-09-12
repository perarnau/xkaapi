#!/usr/bin/env sh

XKAAPIDIR=$HOME/install/xkaapi/default
CUDADIR=$CUDA_HOME

$CUDADIR/bin/nvcc -w -g \
    -I$XKAAPIDIR/include \
    -I$CUDADIR/include \
    -DKAAPI_NDEBUG=1 \
    -DKAAPI_DEBUG=0 \
    transform.cu \
    -o transform \
    -L$XKAAPIDIR/lib -lkaapi -lkaapi++ \
    -L$CUDADIR/lib64 -lcuda -lcublas
