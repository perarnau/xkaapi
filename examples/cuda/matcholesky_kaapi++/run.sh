#!/bin/bash

SCRATCH=/scratch/jvlima
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SCRATCH/install/xkaapi/default/lib:$LD_LIBRARY_PATH

version="$(date +%s)"
out="$SCRATCH/res/xkaapi-spotrf-4cpu1gpu-${version}.txt"

function run_test {
    export KAAPI_CPUSET="0:2"
    #export KAAPI_GPUSET="0~6,1~7"
    export KAAPI_GPUSET="0~6"
#    msizes="10240"
    msizes="4096"
#    bsizes="1024"
#    bsizes="2048"
    bsizes="512"
    niter=1
    verif=1
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET \
		    ./matcholesky_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE=260046848 ./matcholesky_kaapi++ $m $b 1 $verif 
	    #	KAAPI_STACKSIZE=536870912 ./matprod_iter_kaapi++ $m $b $verif
	    #	KAAPI_STACKSIZE=536870912 gdb ./matprod_iter_kaapi++ 
	    done
	done
    done
}

run_test

exit 0

function run_potrf {
    ncpu=6
    ngpu="$1"
    gpuset="$2"

    out="$SCRATCH/res/xkaapi-spotrf-${ncpu}cpu${ngpu}gpu-${version}.txt"
    export KAAPI_GPUSET="$gpuset"
    export KAAPI_CPUSET='0:5'
    msizes="1024 2048"
    bsizes="512"
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET \
		    ./matcholesky_kaapi++ $m $b $verif >> $out"
	    KAAPI_STACKSIZE=260046848 ./matcholesky_kaapi++ $m $b 1 $verif 
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
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET \
		    ./matcholesky_kaapi++ $m $b $verif >> $out"
	    KAAPI_STACKSIZE=260046848 ./matcholesky_kaapi++ $m $b 1 $verif 
	    done
	done
    done
}

GPUSET='3~7 4~8 5~9 6~10 7~11'
init_gpuset="0~6"
let 'ngpu=1'
run_potrf "$ngpu" "$init_gpuset"
for gset in $GPUSET
do
    init_gpuset="$init_gpuset,$gset"
    let 'ngpu++'
    run_potrf "$ngpu" "$init_gpuset"
done

