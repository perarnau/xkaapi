#!/usr/bin/env sh
export LD_LIBRARY_PATH=/usr/local/stow/cuda-4.0/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/lementec/install/xkaapi_gpu/lib:$LD_LIBRARY_PATH

# CONFIG_USE_STATIC == 1
#export KAAPI_CPUSET=0,1
#export KAAPI_GPUSET='0~2,1~3'

#export KAAPI_CPUSET=0:7
#export KAAPI_GPUSET='0~14,1~15'
#export KAAPI_GPUSET='0~14'

# export KAAPI_CPUSET=0
export KAAPI_CPUSET=
export KAAPI_GPUSET='0~0'


echo -n '# blocSize' ;
for m in 64 128 256 512 1024 2048 ; do
    echo -n " m=$m" ;
done
echo ;

for b in 1 2 4 8 16 32 64; do
    echo -n $b;
    for m in 64 128 256 512 1024 2048 ; do
	echo -n ' ' ;
	KAAPI_STACKSIZE=260046848 ./a.out $m $b ;
    done
    echo ;
done
