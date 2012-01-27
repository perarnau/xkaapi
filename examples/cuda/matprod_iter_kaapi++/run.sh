#!/usr/bin/env sh

export LD_LIBRARY_PATH=$CUDA_HOME/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/install/xkaapi/default/lib:$LD_LIBRARY_PATH

niter=1
#niter=30
version="$(date +%s)"
out="$HOME/res/xkaapi-sgemm-4cpu1gpu-${version}.txt"

#export KAAPI_CPUSET='0:3'
export KAAPI_CPUSET='0:1'
#export KAAPI_CPUSET='0'

export KAAPI_GPUSET='0~1'
#export KAAPI_GPUSET=''
#export KAAPI_GPUSET='0~2,1~3'

#export KAAPI_DUMP_GRAPH='1'

verif="1"

# small test configuration
# bsizes="1" ;
# msizes="32" ;

# volkov configuration
#bsizes="2 4 8 16" ;
# msizes="2048" ;
# bsizes="1 2 4 8 16 32 64 128" ;
# msizes="4096" ;
# bsizes="1 2 4 8" ;
# msizes="8192" ;

# bsizes="1 2 4 8" ;
# msizes="16384" ;

# test configuration
#bsizes="16"
#bsizes="2"
#bsizes="1 2 4 8 16 32 64"
bsizes="4"
#bsizes="16"
#msizes="2048"
#msizes="64"
#msizes="128"
#msizes="256"
msizes="512"
#msizes="1024"
#msizes="2048"
#msizes="4096"
#msizes="8192"
#msizes="10240"
#msizes="$(seq 64 64 2048) $(seq 3072 1024 10240)"
#msizes="12288" # ouch
#msizes="16384" # ouch
#msizes="20480"
# bsizes="2 4 8 16"

# bsizes="1 2 4 8 16 32" ;
# msizes="32 64 256" ;

#echo -n '# blocSize' ;
#for m in $msizes ; do
#    echo -n " m=$m" ;
#done
#echo ;

for m in $msizes ; do
	for b in $bsizes; do
#	echo "GEMM $m $b"
	for i in `seq 1 $niter`
	do
	#KAAPI_STACKSIZE=260046848 ./matprod_iter_kaapi++ $m $b $verif >> $out
	KAAPI_STACKSIZE=260046848 ./matprod_iter_kaapi++ $m $b $verif
#	KAAPI_STACKSIZE=260046848 gdb ./matprod_iter_kaapi++
	done
    done
done
