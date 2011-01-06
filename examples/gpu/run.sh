#!/usr/bin/env sh
export LD_LIBRARY_PATH=/usr/local/stow/cuda-3.2/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/lementec/install/xkaapi_gpu/lib:$LD_LIBRARY_PATH
#export KAAPI_CPUSET=0:7
#export KAAPI_GPUSET='0~8,1~15'
export KAAPI_CPUSET=0
export KAAPI_GPUSET='0~1'
./a.out
