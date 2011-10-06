#!/usr/bin/env bash

KAAPI_HWS_LEVELS=FLAT \
KAAPI_EMITSTEAL=hws \
KAAPI_CPUSET=0:3 \
KAAPI_STACKSIZE=536870912 \
LD_LIBRARY_PATH=$HOME/install/xkaapi_hws/lib \
gdb ./a.out
