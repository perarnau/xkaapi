#!/usr/bin/env sh

# globals
export LD_LIBRARY_PATH=$HOME/install/lib
DIR=/tmp/out
SIZE=100000
ITER=10
#CPUSET=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
#CPUSET=8,9,10,11,12,13,14,15
#CPUSET=0,1,2,3,4,5,6,7
#CPUSET=5,7,9,11
#CPUSET=7,9,11
#CPUSET=9,11
#CPUSET=9
CPUSET=0,1
#CPUSET=1
ALGOS=min_element
LIBS=kastl
DOS=check


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
    DO_ALGOS=0 ;
    DO_LIBS=0 ;
    DO_DOS=0 ;

    [ $ALGOS'' == '' ] && DO_ALGOS=1 ;
    [ $LIBS'' == '' ] && DO_LIBS=1 ;
    [ $DOS'' == '' ] && DO_DOS=1 ;

    for f in $1/*; do
	[ -e $f ] || continue ;

	s=`basename $f`
	s=(${s//-/ })

	if [ $DO_ALGOS -eq 1 ]; then
	    a=${s[0]}
	    is_in_list "$ALGOS" $a ;
	    if [ $? -eq 0 ] ; then
		ALGOS="$ALGOS $a" ;
	    fi
	fi

	if [ $DO_LIBS -eq 1 ]; then
	    l=${s[1]}
	    is_in_list "$LIBS" $l ;
	    if [ $? -eq 0 ] ; then
		LIBS="$LIBS $l" ;
	    fi
	fi

	if [ $DO_DOS -eq 1 ]; then
	    d=${s[2]}
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
    rm -rf $DIR
    mkdir $DIR

    print_lists ;

    for ALGO in $ALGOS; do
	for DO in $DOS; do
	    for LIB in $LIBS; do
		NAME=$ALGO-$LIB-$DO
		if [ -e ../bin/$NAME ]; then
		    OUTPUT_FILE=/tmp/oo/$NAME.o ;
		    KAAPI_CPUSET=$CPUSET ../bin/$NAME $SIZE $ITER > $OUTPUT_FILE ;
		    echo == ;
		fi
	    done
	done
    done
}


main() {
    [ -d $1 ] || exit -1 ;

    gen_lists $1;
    run_lists ;
}

main $@;
