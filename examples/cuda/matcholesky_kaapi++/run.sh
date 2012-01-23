#!/bin/bash

export LD_LIBRARY_PATH=$CUDA_HOME/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/install/xkaapi/mat_gpu/lib:$LD_LIBRARY_PATH

#niter=30
niter=1
version="$(date +%s)"
out="$HOME/res/xkaapi-spotrf-4cpu1gpu-${version}.txt"

# CONFIG_USE_STATIC == 1
#export KAAPI_CPUSET=0,1
#export KAAPI_GPUSET='0~2,1~3'

#export KAAPI_CPUSET=0:7
#export KAAPI_GPUSET='0~14,1~15'
#export KAAPI_GPUSET='0~14'

export KAAPI_CPUSET='0:2'
#export KAAPI_CPUSET='0'

#export KAAPI_GPUSET=''
export KAAPI_GPUSET='0~7'
#export KAAPI_GPUSET='0~6,1~7'

#msize="512"
#msize="64"
#msize="2048"
msizes="4096"
#msize="8192"
#msizes="$(seq 64 64 2048) $(seq 3072 1024 10240)"
#msizes="1024"
#nb="2 4 8 12 16 32"
#nb="32"
nb="16"
verif="1"

for m in $msizes 
do
for b in $nb 
do
#    echo "# SPOTRF $m $b"
    for i in `seq 1 $niter`
    do
	#KAAPI_STACKSIZE=260046848 ./matcholesky_kaapi++ $m $b 1 $verif >> $out
	KAAPI_STACKSIZE=260046848 ./matcholesky_kaapi++ $m $b 1 $verif 
    done
done
done
