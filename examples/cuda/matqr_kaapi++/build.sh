#!/bin/bash

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

do_test "CBLAS_CFLAGS" "No CBLAS_CFLAGS found."
do_test "CBLAS_LDFLAGS" "No CBLAS_LDFLAGS found."
#do_test "LAPACK_CFLAGS" "No LAPACK_CFLAGS found."
#do_test "LAPACK_LDFLAGS" "No LAPACK_LDFLAGS found."
do_test "LAPACKE_CFLAGS" "No LAPACKE_CFLAGS found."
do_test "LAPACKE_LDFLAGS" "No LAPACKE_LDFLAGS found."
do_test "PLASMA_CFLAGS" "No LAPACKE_LDFLAGS found."
do_test "PLASMA_LDFLAGS" "No LAPACKE_LDFLAGS found."

#CUBLAS_CFLAGS="-DCONFIG_USE_CUBLAS=1"
#CUBLAS_LDFLAGS="-lcublas"

#CUDA_CFLAGS="-DCONFIG_USE_CUDA=1 $CUDA_CFLAGS"

CFLAGS="-DKAAPI_DEBUG=0 -DKAAPI_NDEBUG=1 
-DCONFIG_USE_DOUBLE=1 -I$XKAAPIDIR/include "
LDFLAGS="-L$XKAAPIDIR/lib -lkaapi -lkaapi++ -lgfortran"

PLASMA_CFLAGS="-DCONFIG_USE_PLASMA=1 $PLASMA_CFLAGS"

#MAGMA_CFLAGS="-DCONFIG_USE_MAGMA=1 $MAGMA_CFLAGS"

$CXX -O3 -Wall \
    $CFLAGS \
    $CUDA_CFLAGS \
    $CBLAS_CFLAGS \
    $CUBLAS_CFLAGS \
    $LAPACK_CLAGS \
    $LAPACKE_CFLAGS \
    $MAGMA_CFLAGS \
    $PLASMA_CFLAGS \
    -c matqr_kaapi++.cpp 

$CXX -O3 \
    -o matqr_kaapi++ \
    matqr_kaapi++.o \
    $LDFLAGS \
    $MAGMA_LDFLAGS \
    $PLASMA_LDFLAGS \
    $CUDA_LDFLAGS \
    $CUBLAS_LDFLAGS \
    $LAPACKE_LDFLAGS \
    $LAPACK_LDFLAGS \
    $CBLAS_LDFLAGS 
