#!/usr/bin/env sh

# globals
export LD_LIBRARY_PATH=$HOME/install/lib
SESSION_DIR=

# input sizes
#SIZE0=1000
#SIZE0=2000
SIZE1=5000
SIZE2=20000
#SIZE3=50000
SIZE4=100000
#SIZE5=200000
#SIZE6=1000000
#SIZE7=
#SIZE8=

# iteration
ITER=100

# cpuset
CPUSET0=1
#CPUSET1=0,1
#CPUSET0=9
CPUSET1=9,11
#CPUSET2=7,9,11
CPUSET3=5,7,9,11
CPUSET4=0,1,2,3,4,5,6,7
#CPUSET5=8,9,10,11,12,13,14,15
#CPUSET6=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
#CPUSET6=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
#CPUSET7=
#CPUSET8=

# {algo, lib, do}
#ALGOS='min_element max_element for_each count inner_product'
#ALGOS='accumulate'
#ALGOS='for_each'
#ALGOS='max_element'
ALGOS="ALGOS find_first_of"
LIBS='kastl'
DOS='check'


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
		    echo $SESSION_DIR/$NAME-$SIZE-$CPUSETID ;
		    OUTPUT_FILE=$SESSION_DIR/$NAME-$SIZE-$CPUSETID ;
		    KAAPI_CPUSET=$CPUSET ../bin/$NAME $SIZE $ITER > $OUTPUT_FILE ;
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
