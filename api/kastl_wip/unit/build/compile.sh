#!/usr/bin/env sh

XKAAPI_INSTALL_DIR=$HOME/install
TBB_INSTALL_DIR=$HOME/install
KASTL_INSTALL_DIR=`pwd`/..

rm main

g++ \
    -DKASTL_DEBUG=0 -std=gnu++0x -Wall -O3 \
    -fopenmp \
    -I$XKAAPI_INSTALL_DIR/include \
    -I$TBB_INSTALL_DIR/include \
    -I$KASTL_INSTALL_DIR \
    -o main \
    -DCONFIG_ALGO_FOR_EACH=1 \
    -DCONFIG_LIB_PASTL=1 \
    -DCONFIG_DO_BENCH=1 \
    main.cc \
    $KASTL_INSTALL_DIR/kastl/kastl_workqueue.cpp \
    -L$TBB_INSTALL_DIR/lib \
    -L$XKAAPI_INSTALL_DIR/lib
#    -lomp
#    -lpthread -ltbb -lxkaapi