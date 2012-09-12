#!/bin/bash

SCRATCH=/tmp
#XKAAPIDIR=/tmp/xkaapi
XKAAPIDIR=$HOME/install/xkaapi/default

CXX=g++

function do_test() {
    eval var=\$$1
    if [ "x$var" = "x" ]
    then
        echo "ERROR: $2"
        exit 0
    fi
}

#CBLAS_CFLAGS="-I/usr/local/atlas/include"
#CBLAS_LDFLAGS="/usr/local/atlas/lib/libcblas.a /usr/local/atlas/lib/liblapack.a /usr/local/atlas/lib/libatlas.a"
#LAPACKE_CFLAGS="-I/usr/local/include"
#LAPACKE_LDFLAGS="-L/usr/local/lib -llapacke -llapack"

do_test "CBLAS_CFLAGS" "No CBLAS_CFLAGS found."
do_test "CBLAS_LDFLAGS" "No CBLAS_LDFLAGS found."
#do_test "LAPACK_CFLAGS" "No LAPACK_CFLAGS found."
#do_test "LAPACK_LDFLAGS" "No LAPACK_LDFLAGS found."
do_test "LAPACKE_CFLAGS" "No LAPACKE_CFLAGS found."
do_test "LAPACKE_LDFLAGS" "No LAPACKE_LDFLAGS found."
#do_test "CUDA_CFLAGS" "No CUDA_CFLAGS found."
#do_test "CUDA_LDFLAGS" "No CUDA_LDFLAGS found."

CFLAGS="-DKAAPI_DEBUG=0 -DKAAPI_NDEBUG=1 
-DCONFIG_USE_DOUBLE=1 -I$XKAAPIDIR/include"
LDFLAGS="-L$XKAAPIDIR/lib -lkaapi -lkaapi++"


CUDA_CFLAGS="-DCONFIG_USE_CUDA=1 $CUDA_CFLAGS"
CUBLAS_CFLAGS="-DCONFIG_USE_CUBLAS=1"
CUBLAS_LDFLAGS="-lcublas"

#g++ -g -Wall 
$CXX -O3 -Wall \
    $CFLAGS \
    $CUDA_CFLAGS \
    $CBLAS_CFLAGS \
    $CUBLAS_CFLAGS \
    $LAPACK_CLAGS \
    $LAPACKE_CFLAGS \
    $MAGMA_CFLAGS \
    -o matprod_rec_kaapi++ \
    matprod_rec_kaapi++.cpp \
    $LDFLAGS \
    $CUDA_LDFLAGS \
    $CUBLAS_LDFLAGS \
    $CBLAS_LDFLAGS \
    $LAPACKE_LDFLAGS \
    $LAPACK_LDFLAGS \
    $MAGMA_LDFLAGS

