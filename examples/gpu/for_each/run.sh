#!/usr/bin/env sh
export LD_LIBRARY_PATH=/usr/local/stow/cuda-4.0/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/install/xkaapi_gpu/lib:$LD_LIBRARY_PATH

#export KAAPI_CPUSET=0
export KAAPI_CPUSET=0
export KAAPI_GPUSET='1~1'

KAAPI_STACKSIZE=260046848 ./a.out 1048576
