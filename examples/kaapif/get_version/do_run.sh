#!/usr/bin/env sh

HOME=/home/lementec

export LD_LIBRARY_PATH=$HOME/install/xkaapi_master/lib ;
export KAAPI_STACKSIZE=536870912;
export KAAPI_CPUSET=0
./a.out ;
