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
#do_test "CUDA_CFLAGS" "No CUDA_CFLAGS found."
#do_test "CUDA_LDFLAGS" "No CUDA_LDFLAGS found."
#do_test "MAGMA_CFLAGS" "No MAGMA_CFLAGS found."
#do_test "MAGMA_LDFLAGS" "No MAGMA_LDFLAGS found."

CFLAGS="-DCONFIG_USE_FLOAT=1 -I$XKAAPIDIR/include"
LDFLAGS="-L$XKAAPIDIR/lib -lkaapi -lkaapi++"

CUDA_CFLAGS="-DCONFIG_USE_CUDA=1 $CUDA_CFLAGS"
CUBLAS_CFLAGS="-DCONFIG_USE_CUBLAS=1"
CUBLAS_LDFLAGS="-lcublas"

#MAGMA_CFLAGS="-DCONFIG_USE_MAGMA=1 $MAGMA_CFLAGS"

$CXX -g -Wall \
    $CFLAGS \
    $CUDA_CFLAGS \
    $CBLAS_CFLAGS \
    $CUBLAS_CFLAGS \
    $LAPACKE_CFLAGS \
    $LAPACK_CFLAGS \
    -c matlu_nopiv_kaapi++.cpp 


$CXX -g \
    -o matlu_nopiv_kaapi++ \
    matlu_nopiv_kaapi++.o \
    $LDFLAGS \
    $CUDA_LDFLAGS \
    $CUBLAS_LDFLAGS \
    $LAPACKE_LDFLAGS \
    $LAPACK_LDFLAGS \
    $CBLAS_LDFLAGS 

