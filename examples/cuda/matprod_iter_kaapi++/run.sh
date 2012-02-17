#!/bin/bash

#SCRATCH=/scratch/jvlima
SCRATCH=$HOME
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SCRATCH/install/xkaapi/default/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$SCRATCH/adonis/xkaapi/default/lib:$LD_LIBRARY_PATH

niter=1
#niter=30
version="$(date +%s)"

#export KAAPI_DUMP_GRAPH='1'

function run_test {
    export KAAPI_CPUSET="0:1"
#    export KAAPI_GPUSET=""
#    export KAAPI_GPUSET="0~7"
#    export KAAPI_GPUSET="0~6,3~7"
    export KAAPI_GPUSET="0~6,1~7"
#    msizes="10240"
#    msizes="2048"
#    msizes="16384"
    msizes="2048"
    bsizes="512"
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
#	    	KAAPI_STACKSIZE=536870912 ./matprod_iter_kaapi++ $m $b $verif
#       KAAPI_STACKSIZE=536870912 gdb ./matprod_iter_kaapi++ 
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

