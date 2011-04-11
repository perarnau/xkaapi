#!/usr/bin/env sh

XKAAPIDIR=$HOME/install/xkaapi_gpu
CUDADIR=/usr/local/stow/cuda-4.0/cuda

# use this flag for atomic instruction 
# does not work on gtx280
# -arch=compute_20

$CUDADIR/bin/nvcc -w \
    -I$XKAAPIDIR/include \
    -I$CUDADIR/include \
    scale_image_kaapi++.cu \
    -L$XKAAPIDIR/lib -lkaapi -lkaapi++ \
    -L$CUDADIR/lib64 -lcuda
