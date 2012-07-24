#!/bin/bash

SCRATCH=$SCRATCH
CUDADIR=$SCRATCH/install/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$SCRATCH/install/xkaapi/default/lib:$LD_LIBRARY_PATH

version="$(date +%s)"

function run_test {
#    export KAAPI_CPUSET="0"
    export KAAPI_CPUSET="10"
#    export KAAPI_CPUSET="4,5,10,11"
#    export KAAPI_GPUSET="0~0"
#    export KAAPI_GPUSET="0~0,1~1"
#    export KAAPI_GPUSET="0~0,1~1,2~2,3~3,4~6,5~7,6~8,7~9"
#    export KAAPI_GPUSET="0~0,2~2,4~6,6~8"
    export KAAPI_GPUSET="4~6,5~7,6~8,7~9"
#    export KAAPI_GPUSET="0~0,1~1,2~2,3~3"
#    export KAAPI_GPUSET="4~6"

    export MKL_NUM_THREADS=2

#    export COMPUTE_PROFILE=1
#    export COMPUTE_PROFILE_CSV=1
#    export COMPUTE_PROFILE_CONFIG="$HOME/compute_profile_config.txt"

#    export KAAPI_RECORD_TRACE=1
#    export KAAPI_RECORD_MASK="COMPUTE,IDLE"

#    export KAAPI_DISPLAY_PERF=1
#    export KAAPI_AFFINITY=1

#    export KAAPI_DUMP_GRAPH=1
#    export KAAPI_DOT_NOVERSION_LINK=1
#    export KAAPI_DOT_NODATA_LINK=1
#    export KAAPI_DOT_NOLABEL_INFO=1
#    export KAAPI_DOT_NOACTIVATION_LINK=1
#    export KAAPI_DOT_NOLABEL=1

#    msizes="$(seq 512 512 2048)"
    msizes="16384"
#    msizes="8192"
#    msizes="4096"
#    msizes="20480"
    bsizes="1024"
#    bsizes="2048"
#    bsizes="512"

    niter=2
#    verif=1
#    nwindow="1 2 4 8 12 16 32"
    nwindow="2"
    for w in $nwindow
    do
    export KAAPI_WINDOW_SIZE=$w
    echo "$w"
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
#	    echo "$KAAPI_CPUSET $KAAPI_GPUSET ./matcholesky_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE=536870912 ./matcholesky_kaapi++ $m $b 1 $verif 
#	    KAAPI_STACKSIZE=536870912 gdb ./matcholesky_kaapi++ 
	    done
	done
    done
    done
}

run_test
exit 0

function run_potrf {
    ncpu="$1"
    cpuset="$2"
    ngpu="$3"
    gpuset="$4"

    export KAAPI_GPUSET="$gpuset"
    export KAAPI_CPUSET="$cpuset"
#    msizes="256 $(seq 512 512 2048)"
#    msizes="256 512 1024 2048 $(seq 4096 2048 20480)"
#    msizes="$(seq 20480 2048 30720)"
    msizes="256 512 1024 2048 $(seq 4096 2048 30720)"
    bsizes="1024"
    niter=30
    wins="2"
    for w in $wins
    do
	export KAAPI_WINDOW_SIZE=$w
	out="$HOME/res/xkaapi-dpotrf-inf-prio-gpu-${ncpu}cpu${ngpu}gpu-${version}.txt"
    for m in $msizes
    do
	    for b in $bsizes
	    do
		if [ $m -le 2048 ]
		then
		    b=512
		fi
		if [ $m -eq 256 ]
		then
		    b=128
		fi
		if [ $m -eq 512 ]
		then
		    b=256
		fi
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET ./matcholesky_kaapi++ $m $b $verif >> $out"
	    KAAPI_STACKSIZE=536870912  ./matcholesky_kaapi++ $m $b 1 $verif >> $out
	    done
	done
    done
    done
}

ncpu=1
cpuset="4"

ngpu=1
gpuset="0~0"
run_potrf "$ncpu" "$cpuset" "$ngpu" "$gpuset"

exit 0

ngpu=2
gpuset="0~0,2~2"
run_potrf "$ncpu" "$cpuset" "$ngpu" "$gpuset"

ngpu=4
gpuset="0~0,2~2,4~6,6~8"
run_potrf "$ncpu" "$cpuset" "$ngpu" "$gpuset"

ngpu=8
gpuset="0~0,1~1,2~2,3~3,4~6,5~7,6~8,7~9"
run_potrf "$ncpu" "$cpuset" "$ngpu" "$gpuset"
