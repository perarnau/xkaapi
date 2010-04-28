#!/usr/bin/env sh

#XKAAPI_INSTALL_DIR=`pwd`/../install
XKAAPI_INSTALL_DIR=$HOME/install
TBB_INSTALL_DIR=$HOME/install
KASTL_INSTALL_DIR=`pwd`/..

echo WARNING, check O2
rm main

g++ \
    -DKASTL_DEBUG=0 -std=gnu++0x -Wall -O3 \
    -I$XKAAPI_INSTALL_DIR/include \
    -I$TBB_INSTALL_DIR/include \
    -I$KASTL_INSTALL_DIR \
    -o main \
    main.cc \
    $KASTL_INSTALL_DIR/kastl/kastl_workqueue.cpp \
    -L$TBB_INSTALL_DIR/lib \
    -L$XKAAPI_INSTALL_DIR/lib \
    -lxkaapi -lpthread -ltbb
