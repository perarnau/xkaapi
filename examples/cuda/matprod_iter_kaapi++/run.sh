#!/usr/bin/env sh

SCRATCH=/scratch/jvlima
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SCRATCH/install/xkaapi/default/lib:$LD_LIBRARY_PATH

niter=1
#niter=30
version="$(date +%s)"
out="$SCRATCH/res/xkaapi-sgemm-4cpu1gpu-${version}.txt"

#export KAAPI_CPUSET='0:3'
#export KAAPI_CPUSET='0:2'
#export KAAPI_CPUSET='0:11'
export KAAPI_CPUSET='0:5'

#export KAAPI_GPUSET='0~6'
#export KAAPI_GPUSET='1~2'
export KAAPI_GPUSET='0~6,3~7,4~8,5~9'
#export KAAPI_GPUSET='0~6,3~7'

#export KAAPI_DUMP_GRAPH='1'

#verif="1"

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
#bsizes="64"
#bsizes="128"
#bsizes="256"
#bsizes="512"
bsizes="1024"
#bsizes="2048"


#msizes="2048"
#msizes="128"
#msizes="256"
#msizes="512"
#msizes="1024"
#msizes="2048"
#msizes="4096"
#msizes="8192"
msizes="10240"
#msizes="$(seq 64 64 2048) $(seq 3072 1024 10240)"
#msizes="12288" # ouch
#msizes="16384" # ouch
#msizes="20480"
#msizes="40960"
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
#	KAAPI_STACKSIZE=260046848 ./matprod_iter_kaapi++ $m $b $verif
	KAAPI_STACKSIZE=536870912 ./matprod_iter_kaapi++ $m $b $verif
#	KAAPI_STACKSIZE=67108864 ./matprod_iter_kaapi++ $m $b $verif
#	KAAPI_STACKSIZE=536870912 gdb ./matprod_iter_kaapi++ 
	done
    done
done
