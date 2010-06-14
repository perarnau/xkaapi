#!/usr/bin/env sh

# globals
export LD_LIBRARY_PATH=$HOME/install/lib
SESSION_DIR=

# input sizes
#SIZE0=1000
#SIZE0=1000
#SIZE1=5000
#SIZE2=10000
#SIZE3=20000
#SIZE4=50000
SIZE5=4000000
#SIZE6=200000
#SIZE7=500000
#SIZE8=1000000

#SIZE0=60000
#SIZE1=61000
#SIZE2=62000
#SIZE3=63000
#SIZE4=64000
#SIZE5=65000
#SIZE6=66000
#SIZE7=67000
#SIZE8=68000
#SIZE4=100000
#SIZE5=130000
#SIZE6=150000
#SIZE7=200000
#SIZE8=260000
#SIZE8=1000000

# iteration
#ITER=1
ITER=10
#ITER=20
#ITER=1000

# cpuset
#CPUSET0=2
#CPUSET0=1
#CPUSET1=0,1
#CPUSET0=9
#CPUSET0=12
#CPUSET1=12,13
#CPUSET2=12,14
#CPUSET1=8,9
#CPUSET2=7,9,11
#CPUSET2=12,13,14,15
#CPUSET4=0,1,2,3,4,5,6,7
#CPUSET3=8,9,10,11,12,13,14,15
#CPUSET4=4,5,6,7,8,9,10,11,12,13,14,15
CPUSET5=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
#CPUSET7=
#CPUSET8=

# {algo, lib, do}
ALGOS="$ALGOS transform"
# ALGOS="$ALGOS for_each"
# ALGOS="$ALGOS find"
# ALGOS="$ALGOS find_if"
# ALGOS="$ALGOS find_first_of"
# ALGOS="$ALGOS accumulate"
# ALGOS="$ALGOS inner_product"
# ALGOS="$ALGOS count"
# ALGOS="$ALGOS count_if"
# ALGOS="$ALGOS copy"
# ALGOS="$ALGOS search"
# ALGOS="$ALGOS min_element"
# ALGOS="$ALGOS max_element"
# ALGOS="$ALGOS fill"
# ALGOS="$ALGOS generate"
# ALGOS="$ALGOS inner_product"
# ALGOS="$ALGOS replace"
# ALGOS="$ALGOS replace_if"
# ALGOS="$ALGOS equal"
# ALGOS="$ALGOS mismatch"
# ALGOS="$ALGOS search"
# ALGOS="$ALGOS adjacent_find"
# ALGOS="$ALGOS adjacent_difference"
# ALGOS="$ALGOS partial_sum"

LIBS=''
LIBS="$LIBS kastl"
#LIBS="$LIBS stl"
#LIBS="$LIBS tbb"
#LIBS="$LIBS pastl"
#LIBS="$LIBS pthread"

DOS=''
DOS="$DOS bench"
#DOS="$DOS check"


gen_session_dir() {
    if [ $SESSION_DIR'' == '' ] ; then
	local h=`date +%S%j%Y` ;
	export SESSION_DIR=../session/$h ;
    fi

    [ -d $SESSION_DIR ] || mkdir -p $SESSION_DIR

    [ -h ../session/this ] && unlink ../session/this
    ln -s $SESSION_DIR ../session/this
}


is_in_list() {
    # $1 the list
    # $2 the item

    for i in `echo $1 | cut -d\  -f1-`; do
	if [ $i == $2 ] ; then
	    return 1;
	fi
    done

    return 0;
}


gen_lists() {
    local DO_ALGOS=0 ;
    local DO_LIBS=0 ;
    local DO_DOS=0 ;

    [ "$ALGOS"'' == '' ] && DO_ALGOS=1 ;
    [ "$LIBS"'' == '' ] && DO_LIBS=1 ;
    [ "$DOS"'' == '' ] && DO_DOS=1 ;

    for f in $1/*; do
	[ -e $f ] || continue ;

	s=`basename $f`
	s=(${s//-/ })

	if [ $DO_ALGOS -eq 1 ]; then
	    local a=${s[0]}
	    is_in_list "$ALGOS" $a ;
	    if [ $? -eq 0 ] ; then
		ALGOS="$ALGOS $a" ;
	    fi
	fi

	if [ $DO_LIBS -eq 1 ]; then
	    local l=${s[1]}
	    is_in_list "$LIBS" $l ;
	    if [ $? -eq 0 ] ; then
		LIBS="$LIBS $l" ;
	    fi
	fi

	if [ $DO_DOS -eq 1 ]; then
	    local d=${s[2]}
	    is_in_list "$DOS" $d ;
	    if [ $? -eq 0 ] ; then
		DOS="$DOS $d" ;
	    fi
	fi
    done
}


print_lists() {
    echo $ALGOS ;
    echo $LIBS ;
    echo $DOS ;
}


run_lists() {
    # global variables that need to be defined
    # SIZE, CPUSET, CPUSETID
    for ALGO in $ALGOS; do
	for DO in $DOS; do
	    for LIB in $LIBS; do
		NAME=$ALGO-$LIB-$DO
		if [ -e ../bin/$NAME ]; then
		    export KAAPI_WSSELECT="workload"
		    echo $SESSION_DIR/$NAME-$SIZE-$CPUSETID ;
		    OUTPUT_FILE=$SESSION_DIR/$NAME-$SIZE-$CPUSETID ;
		    KAAPI_CPUSET=$CPUSET numactl -C $CPUSET ../bin/$NAME $SIZE $ITER ;
		fi
	    done
	done
    done
}


run_cpuset() {
    [ -z "$1" ] && return ;
    CPUSET=$1 ;
    CPUSETID=$2 ;
    run_lists ;
}


run_cpusets() {
    run_cpuset "$CPUSET0" '0' ;
    run_cpuset "$CPUSET1" '1' ;
    run_cpuset "$CPUSET2" '2' ;
    run_cpuset "$CPUSET3" '3' ;
    run_cpuset "$CPUSET4" '4' ;
    run_cpuset "$CPUSET5" '5' ;
    run_cpuset "$CPUSET6" '6' ;
    run_cpuset "$CPUSET7" '7' ;
    run_cpuset "$CPUSET8" '8' ;
}


run_size() {
    [ -z "$1" ] && return ;
    SIZE=$1 ;
    run_cpusets ;
}


run_sizes() {
    run_size "$SIZE0" '0' ;
    run_size "$SIZE1" '1' ;
    run_size "$SIZE2" '2' ;
    run_size "$SIZE3" '3' ;
    run_size "$SIZE4" '4' ;
    run_size "$SIZE5" '5' ;
    run_size "$SIZE6" '6' ;
    run_size "$SIZE7" '7' ;
    run_size "$SIZE8" '8' ;
}



main() {
    [ -z "$1" ] && exit -1 ;
    [ -d $1 ] || exit -1 ;

    gen_session_dir ;
    gen_lists $1 ;

    run_sizes ;
}

main $@ ;
