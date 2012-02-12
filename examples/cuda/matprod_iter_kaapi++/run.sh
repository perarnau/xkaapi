#!/bin/bash

#SCRATCH=/scratch/jvlima
SCRATCH=$HOME
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SCRATCH/install/xkaapi/default/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$SCRATCH/adonis/xkaapi/default/lib:$LD_LIBRARY_PATH

niter=1
#niter=30
version="$(date +%s)"

#export KAAPI_CPUSET='0:2'
#export KAAPI_CPUSET='0:2'
#export KAAPI_CPUSET='0:11'

#export KAAPI_GPUSET='0~6'
#export KAAPI_GPUSET='0~3'
#export KAAPI_GPUSET='0~6,3~7,4~8,5~9'
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
#bsizes="1024"
#bsizes="2048"

#bsizes="128 256 512 1024 2048"
#bsizes="128 256 512 1024"

#msizes="1024"
#msizes="128"
#msizes="256"
#msizes="512"
#msizes="2048"
#msizes="4096"
#msizes="8192"
#msizes="10240"
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

function run_test {
    export KAAPI_CPUSET="0:2"
    #export KAAPI_GPUSET="0~6,1~7"
    export KAAPI_GPUSET="0~6"
#    msizes="2048"
    msizes="1024"
    bsizes="128"
    niter=1
    verif=1
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET \
		    ./matprod_iter_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE=536870912 \
		    ./matprod_iter_kaapi++ $m $b $verif
	    #	KAAPI_STACKSIZE=536870912 ./matprod_iter_kaapi++ $m $b $verif
	    #	KAAPI_STACKSIZE=536870912 gdb ./matprod_iter_kaapi++ 
	    done
	done
    done
}

run_test

exit 0

function run_sgemm {
    ncpu=6
    ngpu="$1"
    gpuset="$2"

    out="$SCRATCH/res/xkaapi-sgemm-${ncpu}cpu${ngpu}gpu-${version}.txt"
    export KAAPI_GPUSET="$gpuset"
    export KAAPI_CPUSET='0:5'
    msizes="1024 2048"
    bsizes="512"
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET KAAPI_STACKSIZE=536870912 \
		    ./matprod_iter_kaapi++ $m $b $verif >> $out"
	    #	KAAPI_STACKSIZE=536870912 ./matprod_iter_kaapi++ $m $b $verif
	    #	KAAPI_STACKSIZE=536870912 gdb ./matprod_iter_kaapi++ 
	    done
	done
    done
    return
    bsizes="1024"
    msizes="$(seq 3072 1024 40960)"
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
	    KAAPI_STACKSIZE=536870912 ./matprod_iter_kaapi++ $m $b $verif
    #	KAAPI_STACKSIZE=536870912 gdb ./matprod_iter_kaapi++ 
	    done
	done
    done
}

GPUSET='3~7 4~8 5~9 6~10 7~11'
init_gpuset="0~6"
let 'ngpu=1'
run_sgemm "$ngpu" "$init_gpuset"
for gset in $GPUSET
do
    init_gpuset="$init_gpuset,$gset"
    let 'ngpu++'
    run_sgemm "$ngpu" "$init_gpuset"
done

