#!/usr/bin/env sh

LD_LIBRARY_PATH=$HOME/install/xkaapi_release/lib \
    KAAPI_CPUSET='0:3' \
    ./for_each
