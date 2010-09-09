#!/usr/bin/env sh

XKAAPIDIR=$HOME/install/xkaapi_gpu

nvcc --ptx transform_heter.cu

gcc -Wall -std=c99 \
    -I$XKAAPIDIR/include \
    -DKAAPI_NDEBUG=1 \
    -DKAAPI_DEBUG=0 \
    transform_heter.c \
    -L$XKAAPIDIR/lib -lpthread -lxkaapi -lnuma -lcuda
