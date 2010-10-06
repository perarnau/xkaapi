#!/usr/bin/env sh

gcc \
    -std=c99 -Wall -O3 \
    -I/home/lementec/install/xkaapi_release/include \
    -L/home/lementec/install/xkaapi_release/lib \
    -o accumulate \
    accumulate.c \
    -lxkaapi
