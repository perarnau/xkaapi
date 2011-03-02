#!/usr/bin/env sh

gcc -Wall -O3 -march=native \
-I$HOME/install/xkaapi_master/include \
papi_kaapi.c \
-L$HOME/install/lib -lpapi \
-L$HOME/install/xkaapi_master/lib -lkaapi \
-lpthread -lm
