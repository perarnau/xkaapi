#!/bin/bash

XKAAPIDIR=$HOME/install/xkaapi/default
ARCH_GPU="sm_20" # Fermi GPUs
#ARCH_GPU="sm_30" # Kepler GPUs

CFLAGS="-I$XKAAPIDIR/include -DKAAPI_DEBUG=0 -DKAAPI_NDEBUG=1"
LDFLAGS="-L$XKAAPIDIR/lib -lkaapi -lkaapi++ -lcudart"

nvcc -w -g --machine 64 -arch $ARCH_GPU --compiler-options "$CFLAGS" \
    transform_static.cu \
    -o transform_static \
    $LDFLAGS 
