#!/usr/bin/env sh

gcc \
    -std=c99 -Wall -O3 \
    -DCONFIG_CONCURRENT_SPLIT=1 \
    -I/home/lementec/install/xkaapi_release/include \
    -L/home/lementec/install/xkaapi_release/lib \
    -o main_conc_nored \
    main.c \
    -lxkaapi

gcc \
    -std=c99 -Wall -O3 \
    -DCONFIG_CONCURRENT_SPLIT=1 \
    -DCONFIG_REDUCE_RESULT=1 \
    -I/home/lementec/install/xkaapi_release/include \
    -L/home/lementec/install/xkaapi_release/lib \
    -o main_conc_red \
    main.c \
    -lxkaapi
