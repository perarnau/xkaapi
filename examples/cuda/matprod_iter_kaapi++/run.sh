#!/bin/bash

#SCRATCH=/scratch/jvlima
CUDA_HOME=$SCRATCH/install/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SCRATCH/install/xkaapi/default/lib:$LD_LIBRARY_PATH

niter=1
#niter=30
version="$(date +%s)"

#export KAAPI_DUMP_GRAPH='1'

function run_test {
#    export KAAPI_CPUSET="4"
    export KAAPI_CPUSET="0:1"
#    export KAAPI_CPUSET="4,5,10,11"
#    export KAAPI_GPUSET="0~0,2~2,4~6,6~8"
#    export KAAPI_GPUSET="0~0,1~1,2~2,3~3,4~6,5~7,6~8,7~9"
#   export KAAPI_GPUSET="0~0,1~1,2~2,3~3"
#   export KAAPI_GPUSET="4~6,5~7,6~8,7~9"
#   export KAAPI_GPUSET="4~6"

#    export COMPUTE_PROFILE=1
#    export COMPUTE_PROFILE_CSV=1
#    export COMPUTE_PROFILE_CONFIG="$HOME/compute_profile_config.txt"

#    export KAAPI_DUMP_GRAPH=1
#    export KAAPI_DOT_NOVERSION_LINK=1
#    export KAAPI_DOT_NODATA_LINK=1
#    export KAAPI_DOT_NOLABEL_INFO=1
#    export KAAPI_DOT_NOACTIVATION_LINK=1
#    export KAAPI_DOT_NOLABEL=1

#    export KAAPI_RECORD_TRACE=1
#    export KAAPI_RECORD_MASK="COMPUTE,IDLE"

#    export KAAPI_DISPLAY_PERF=1

#    export KAAPI_AFFINITY="writer"
#    export KAAPI_AFFINITY="dw"

#    export KAAPI_CUDA_PEER=1

#  export KAAPI_STEAL_AFFINITY="locality"

#    msizes="20480"
#     msizes="16384"
#    msizes="$(seq 512 512 2048)"
#    msizes="8192"
#    msizes="4096"
    msizes="1024"
#    bsizes="2048"
#    bsizes="1024"
    bsizes="512"
    nwindow="2"
    niter=1
    verif=1
    for w in $nwindow
    do
    export KAAPI_WINDOW_SIZE=$w
    echo "$w"
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
#	    echo "window $KAAPI_WINDOW_SIZE $KAAPI_CPUSET $KAAPI_GPUSET \
#		    ./matprod_iter_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE=536870912 ./matprod_iter_kaapi++ $m $b $verif
#	    KAAPI_STACKSIZE=536870912 gdb ./matprod_iter_kaapi++ 
	    done
	done
    done
    done	
}

run_test
exit 0

function run_dgemm {
    ncpu="$1"
    cpuset="$2"
    ngpu="$3"
    gpuset="$4"

    out="$HOME/res/xkaapi-dgemm-${ncpu}cpu${ngpu}gpu-${version}.txt"
    export KAAPI_GPUSET="$gpuset"
    export KAAPI_CPUSET="$cpuset"
    bsizes="1024"
    msizes="$(seq 4096 1024 20480)"
    niter=30
    for m in $msizes
    do
	    for b in $bsizes
	    do
	    for i in `seq 1 $niter`
	    do
		echo "$KAAPI_CPUSET($ncpu) $KAAPI_GPUSET($ngpu) ./matprod_iter_kaapi++ $m $b $verif"
		KAAPI_STACKSIZE=536870912 ./matprod_iter_kaapi++ $m $b $verif >> $out
	    done
	done
    done
}

function run_dgemm_cpu {
    ncpu="$1"
    cpuset="$2"
    gpuset=""

    out="$HOME/res/xkaapi-dgemm-${ncpu}cpu-nogpu-${version}.txt"
    export KAAPI_GPUSET="$gpuset"
    export KAAPI_CPUSET="$cpuset"
    niter=30

    msizes="$(seq 1024 1024 20480)"
    bsizes="256 512"
    for b in $bsizes; do
	for m in $msizes ; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET($ncpu) ./matprod_iter_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE=536870912 \
		    ./matprod_iter_kaapi++ $m $b $verif >> $out
	    done
	done
    done
    msizes="$(seq 2048 1024 20480)"
    bsizes="1024"
    for b in $bsizes; do
	for m in $msizes ; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET($ncpu) ./matprod_iter_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE=536870912 \
		    ./matprod_iter_kaapi++ $m $b $verif >> $out
	    done
	done
    done
    msizes="128 256 512 $(seq 1024 1024 20480)"
    bsizes="64"
    for b in $bsizes; do
	for m in $msizes ; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET($ncpu) ./matprod_iter_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE=536870912 \
		    ./matprod_iter_kaapi++ $m $b $verif >> $out
	    done
	done
    done
}

function run_test_block {
    ncpu="$1"
    cpuset="$2"
    ngpu="$3"
    gpuset="$4"

    out="$HOME/res/xkaapi-dgemm-${ncpu}cpu${ngpu}gpu-${version}.txt"
    export KAAPI_GPUSET="$gpuset"
    export KAAPI_CPUSET="$cpuset"
    msizes="$(seq 2048 2048 30720)"
    bsizes="2048"
    niter=30
#    verif=1
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET($ncpu) $KAAPI_GPUSET($ngpu) ./matprod_iter_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE=536870912 \
		    ./matprod_iter_kaapi++ $m $b $verif >> $out
#	    	KAAPI_STACKSIZE=536870912 ./matprod_iter_kaapi++ $m $b $verif
#       KAAPI_STACKSIZE=536870912 gdb ./matprod_iter_kaapi++ 
	    done
	done
    done
}

#run_test_block
#exit 0

#ncpu=12
#cpuset="0:11"
#run_dgemm_cpu "$ncpu" "$cpuset" 
#exit 0

ncpu=1
cpuset="4"

ngpu=1
gpuset="0~0"
run_dgemm "$ncpu" "$cpuset" "$ngpu" "$gpuset"

ngpu=2
gpuset="0~0,2~2"
run_dgemm "$ncpu" "$cpuset" "$ngpu" "$gpuset"

ngpu=4
gpuset="0~0,2~2,4~6,6~8"
run_dgemm "$ncpu" "$cpuset" "$ngpu" "$gpuset"

ngpu=8
gpuset="0~0,1~1,2~2,3~3,4~6,5~7,6~8,7~9"
run_dgemm "$ncpu" "$cpuset" "$ngpu" "$gpuset"

# 2 CPU 

ncpu=2
cpuset="4,5"

ngpu=1
gpuset="0~0"
run_dgemm "$ncpu" "$cpuset" "$ngpu" "$gpuset"

ngpu=2
gpuset="0~0,2~2"
run_dgemm "$ncpu" "$cpuset" "$ngpu" "$gpuset"

ngpu=4
gpuset="0~0,2~2,4~6,6~8"
run_dgemm "$ncpu" "$cpuset" "$ngpu" "$gpuset"

ngpu=8
gpuset="0~0,1~1,2~2,3~3,4~6,5~7,6~8,7~9"
run_dgemm "$ncpu" "$cpuset" "$ngpu" "$gpuset"

