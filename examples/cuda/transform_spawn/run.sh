#!/usr/bin/env sh
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/install/xkaapi/default/lib:$LD_LIBRARY_PATH

# CONFIG_USE_STATIC == 1
#export KAAPI_CPUSET=0,1
#export KAAPI_GPUSET='0~2,1~3'

#export KAAPI_CPUSET=0:7
#export KAAPI_GPUSET='0~14,1~15'
#export KAAPI_GPUSET='0~14'

export KAAPI_CPUSET='0:1'
#export KAAPI_CPUSET='0'
export KAAPI_GPUSET='0~3'
#export KAAPI_GPUSET=''
#export KAAPI_DUMP_GRAPH='1'

inputs="1024 2048 4096"

for i in $inputs
do
    KAAPI_STACKSIZE=260046848 ./transform $i
    #KAAPI_STACKSIZE=260046848 gdb ./transform
done
