#!/usr/bin/env sh

XKAAPIDIR=$HOME/install

gcc -Wall -std=c99 \
    -I$XKAAPIDIR/include \
    -DNDEBUG=1 \
    transform_heter.c \
    -L$HOME/install/lib -lpthread -lxkaapi -lnuma -lcuda
