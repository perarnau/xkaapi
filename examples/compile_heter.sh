#!/usr/bin/env sh

XKAAPIDIR=$HOME/install

gcc -Wall -std=c99 \
    -I$XKAAPIDIR/include \
    -DKAAPI_NDEBUG=1 \
    -DKAAPI_DEBUG=0 \
    transform_heter.c \
    -L$HOME/install/lib -lpthread -lxkaapi -lnuma -lcuda
