#!/bin/bash

SCRATCH=$SCRATCH
CUDADIR=$SCRATCH/install/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/install/xkaapi/default/lib:$LD_LIBRARY_PATH

version="$(date +%s)"
dorun="yes"

function run_test {
#    export KAAPI_CPUSET="0"
    export KAAPI_CPUSET="4"
#    export KAAPI_CPUSET="4,5,10,11"
#    export KAAPI_GPUSET="0~0,1~1,2~2,3~3"
    export KAAPI_GPUSET="0~0,1~1,2~2,3~3,4~6,5~7,6~8,7~9"

#    export COMPUTE_PROFILE=1
#    export COMPUTE_PROFILE_CSV=1
#    export COMPUTE_PROFILE_CONFIG="$HOME/compute_profile_config.txt"

#    export KAAPI_RECORD_TRACE=1
#    export KAAPI_RECORD_MASK="COMPUTE,IDLE"

#    export KAAPI_DUMP_GRAPH=1
#    export KAAPI_DOT_NOVERSION_LINK=1
#    export KAAPI_DOT_NODATA_LINK=1
#    export KAAPI_DOT_NOLABEL_INFO=1
#    export KAAPI_DOT_NOACTIVATION_LINK=1
#    export KAAPI_DOT_NOLABEL=1

#    export KAAPI_DISPLAY_PERF=1
#
    export KAAPI_PUSH_AFFINITY="writer"
#    export KAAPI_STEAL_AFFINITY="writer"
#    export KAAPI_PUSH_AFFINITY="locality"
#    export KAAPI_STEAL_AFFINITY="locality"


#    msizes="10240"
#    msizes="256"
    msizes="16384"
    bsizes="1024"
#    bsizes="128"
    niter=1
#    verif=1
    export KAAPI_WINDOW_SIZE=2
    for m in $msizes ; do
	    for b in $bsizes; do
	    for i in `seq 1 $niter`
	    do
	    echo "$KAAPI_CPUSET $KAAPI_GPUSET \
		    ./matcholesky_kaapi++ $m $b $verif"
	    KAAPI_STACKSIZE=536870912 ./matcholesky_kaapi++ $m $b 1 $verif 
#	    KAAPI_STACKSIZE=536870912 gdb ./matcholesky_kaapi++ 
	    done
	done
    done
}

run_test
exit 0

function run_potrf_gpu {
    ncpu="$1"
    cpuset="$2"
    ngpu="$3"
    gpuset="$4"

    export KAAPI_CPUSET="$cpuset"
    export KAAPI_GPUSET="$gpuset"
#    msizes="256 512 1024 2048 $(seq 4096 2048 20480)"
#    msizes="$(seq 22528 2048 20480)"
    msizes="16384"
    bsizes="1024"
    niter=10
    wins="2"
    for w in $wins
    do
	export KAAPI_WINDOW_SIZE=$w
	out="$HOME/res/xkaapi-dpotrf-gpu-${w}w-${ncpu}cpu${ngpu}gpu-${version}.csv"
    for m in $msizes
    do
	    for b in $bsizes
	    do
	    for i in `seq 1 $niter`
	    do
		echo "($i/$niter) $KAAPI_CPUSET $KAAPI_GPUSET ./matcholesky_kaapi++-gpu $m $b $verif >> $out"
		if [ -n "$dorun" ]
		then
		KAAPI_STACKSIZE=536870912  ./matcholesky_kaapi++-gpu $m $b 1 $verif >> $out
		fi
	    done
	done
    done
    done
}

function run_potrf {
    ncpu="$1"
    cpuset="$2"
    ngpu="$3"
    gpuset="$4"

    export KAAPI_CPUSET="$cpuset"
    export KAAPI_GPUSET="$gpuset"
#    msizes="256 512 1024 2048 $(seq 4096 2048 20480)"
#    msizes="$(seq 22528 2048 20480)"
    msizes="8192 10240 16384 20480"
    bsizes="512 1024"
    niter=10
    wins="2"
    exe="./matcholesky_kaapi++-cpu-gpu"
    for w in $wins
    do
	export KAAPI_WINDOW_SIZE=$w
	out="$HOME/res/xkaapi-dpotrf-${w}w-${ncpu}cpu${ngpu}gpu-${version}.csv"
    for m in $msizes
    do
	    for b in $bsizes
	    do
	    for i in `seq 1 $niter`
	    do
		echo "($i/$niter) $KAAPI_CPUSET $KAAPI_GPUSET $exe $m $b $verif >> $out"
		if [ -n "$dorun" ]
		then
		KAAPI_STACKSIZE=536870912 $exe $m $b 1 $verif >> $out
		fi
	    done
	done
    done
    done
}

function run_potrf_mem_gpu {
    ncpu="$1"
    cpuset="$2"
    ngpu="$3"
    gpuset="$4"

    export KAAPI_CPUSET="$cpuset"
    export KAAPI_GPUSET="$gpuset"
#    msizes="256 512 1024 2048 $(seq 4096 2048 20480)"
#    msizes="$(seq 22528 2048 20480)"
    msizes="8192 20480"
    bsizes="1024"
    niter=1
    wins="2"

    export KAAPI_DISPLAY_PERF=1

    for w in $wins
    do
	export KAAPI_WINDOW_SIZE=$w
	out="$HOME/res/memory/xkaapi-dpotrf-memory-${w}w-${ncpu}cpu${ngpu}gpu-${version}.csv"
    for m in $msizes
    do
	    for b in $bsizes
	    do
	    for i in `seq 1 $niter`
	    do
		echo "($i/$niter) $KAAPI_CPUSET $KAAPI_GPUSET ./matcholesky_kaapi++-gpu $m $b $verif >> $out"
		if [ -n "$dorun" ]
		then
		KAAPI_STACKSIZE=536870912  ./matcholesky_kaapi++-gpu $m $b 1 $verif &> $out
		fi
	    done
	done
    done
    done
}

function memory {
    ncpu=1
    cpuset="11"
    export KAAPI_NCPU=$ncpu

    ngpu=8
    export KAAPI_NGPU=$ngpu
    gpuset="0~0,1~1,2~2,3~3,4~6,5~7,6~8,7~9"
    run_potrf_mem_gpu "$ncpu" "$cpuset" "$ngpu" "$gpuset"

    ngpu=6
    export KAAPI_NGPU=$ngpu
    gpuset="0~0,2~2,3~3,4~6,5~7,7~9"
    run_potrf_mem_gpu "$ncpu" "$cpuset" "$ngpu" "$gpuset"

    ngpu=4
    export KAAPI_NGPU=$ngpu
    gpuset="0~0,2~2,4~6,6~8"
    run_potrf_mem_gpu "$ncpu" "$cpuset" "$ngpu" "$gpuset"

    ngpu=2
    export KAAPI_NGPU=$ngpu
    gpuset="4~6,6~8"
    run_potrf_mem_gpu "$ncpu" "$cpuset" "$ngpu" "$gpuset"

    ngpu=1
    export KAAPI_NGPU=$ngpu
    gpuset="6~8"
    run_potrf_mem_gpu "$ncpu" "$cpuset" "$ngpu" "$gpuset"

    return

}

# $1 - (value) ncpus
# $2 - (value) ngpus
# $3 - (value) cpuset
# $4 - (value) gpuset
# $5 - (value) executable file
# $6 - (value) prefix for output
# $7 - (set) block sizes
# $8 - (value) number of repetitions
# $9 - (set) window sizes 
function run_xkaapi_generic {
    ncpu="$1"
    cpuset="$2"
    ngpu="$3"
    gpuset="$4"
    exec_file="$5"
    output_prefix="$6"
    nbsizes="$7"
    niter="$8"
    wins="$9"

    export KAAPI_CPUSET="$cpuset"
    export KAAPI_GPUSET="$gpuset"

#    msizes="$(seq 4096 4096 20480)"
#    msizes="12288 32768"
    msizes="16384"
#    msizes="16384"
#    msizes="$(seq 18432 2048 30720)"
#    msizes="64 128 512 1024 $(seq 2048 2048 20480)"

    for w in $wins
    do
	export KAAPI_WINDOW_SIZE=$w
	out="$HOME/res/${output_prefix}-${w}w${ncpu}cpu${ngpu}gpu-${version}.csv"
	for m in $msizes
	do
	    for b in $nbsizes
	    do
		if [ $m -le $b ]
		then
		    continue
		fi
		for i in `seq 1 $niter`
		do
		    echo "($i/$niter) win=$w $KAAPI_CPUSET($ncpu) $KAAPI_GPUSET($ngpu) $exec_file $m $b $verif >> $out"
		    if [ -n "$dorun" ]
		    then
		    KAAPI_STACKSIZE=536870912 $exec_file $m $b 1 $verif >> $out
		    fi
		done
	    done
	done
    done
}


function hybrid {
    niter=10
    wins="2"
    nblocks="1024"

    ncpu=1
    cpuset="4"
    export KAAPI_NCPU=$ncpu

    ngpu=8
    export KAAPI_NGPU=$ngpu
    gpuset="0~0,1~1,2~2,3~3,4~6,5~7,6~8,7~9"

   export KAAPI_STEAL_AFFINITY="locality"
    run_xkaapi_generic "$ncpu" "$cpuset" "$ngpu" "$gpuset" "./matcholesky_kaapi++" "xkaapiv2-dpotrf-steal-locality" "$nblocks" $niter "$wins"
    export KAAPI_STEAL_AFFINITY="writer"
    run_xkaapi_generic "$ncpu" "$cpuset" "$ngpu" "$gpuset" "./matcholesky_kaapi++" "xkaapiv2-dpotrf-steal-writer" "$nblocks" $niter "$wins"

   unset KAAPI_STEAL_AFFINITY

   export KAAPI_AFFINITY="locality"
    run_xkaapi_generic "$ncpu" "$cpuset" "$ngpu" "$gpuset" "./matcholesky_kaapi++" "xkaapiv2-dpotrf-push-locality" "$nblocks" $niter "$wins"
    export KAAPI_AFFINITY="writer"
    run_xkaapi_generic "$ncpu" "$cpuset" "$ngpu" "$gpuset" "./matcholesky_kaapi++" "xkaapiv2-dpotrf-push-writer" "$nblocks" $niter "$wins"

    exit 0

    ngpu=4
    export KAAPI_NGPU=$ngpu
    gpuset="0~0,1~1,2~2,3~3"

    ngpu=8
    export KAAPI_NGPU=$ngpu
    gpuset="0~0,1~1,2~2,3~3,4~6,5~7,6~8,7~9"
    run_potrf_gpu "$ncpu" "$cpuset" "$ngpu" "$gpuset"

    ngpu=6
    export KAAPI_NGPU=$ngpu
    gpuset="0~0,2~2,3~3,4~6,5~7,7~9"
    run_potrf_gpu "$ncpu" "$cpuset" "$ngpu" "$gpuset"

    return

    ngpu=2
    export KAAPI_NGPU=$ngpu
    gpuset="4~6,6~8"
    run_potrf "$ncpu" "$cpuset" "$ngpu" "$gpuset"

    ngpu=1
    export KAAPI_NGPU=$ngpu
    gpuset="7~9"
    run_potrf "$ncpu" "$cpuset" "$ngpu" "$gpuset"

#    run_potrf_gpu "$ncpu" "$cpuset" "$ngpu" "$gpuset"
#    run_potrf "$ncpu" "$cpuset" "$ngpu" "$gpuset"
#    run_potrf_nothreshold "$ncpu" "$cpuset" "$ngpu" "$gpuset"
#    run_potrf_threshold "$ncpu" "$cpuset" "$ngpu" "$gpuset"
}

hybrid
#memory

exit 0
