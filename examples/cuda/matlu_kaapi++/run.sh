#!/bin/bash

export LD_LIBRARY_PATH=$CUDA_HOME/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/install/xkaapi/mat_gpu/lib:$LD_LIBRARY_PATH

# CONFIG_USE_STATIC == 1
#export KAAPI_CPUSET=0,1
#export KAAPI_GPUSET='0~2,1~3'

#export KAAPI_CPUSET=0:7
#export KAAPI_GPUSET='0~14,1~15'
#export KAAPI_GPUSET='0~14'

export KAAPI_CPUSET='0:1'
#export KAAPI_CPUSET='0'

#export KAAPI_GPUSET=''
export KAAPI_GPUSET='0~2,0~3'

niter=30
#niter=1

#msize="512"
#msize="1024"
#msize="64"
#msize="2048"
msize="4096"

nb="256"
#nb="512"
#nb="1024"
#nb="2048"

#verif="1"

for b in $nb 
do
    for i in `seq 1 $niter`
    do
#	echo "# nblock $b"
	KAAPI_STACKSIZE=260046848 ./matlu_kaapi++ $msize $b 1 $verif 
	#KAAPI_STACKSIZE=260046848 gdb ./matlu_kaapi++ 
    done
done
