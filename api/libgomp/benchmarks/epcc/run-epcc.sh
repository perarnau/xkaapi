#!/bin/bash

SYNCBENCH=$1
shift

REPS=$1
shift

KEYWORDS=
LIBGOMP_PERFS=
LIBKOMP_PERFS=
INTEL_PERFS=
NCORES=

FIRST=0

function parse_log_file {
    if [ $FIRST -eq 0 ];
    then
	KEYWORDS=$(grep "overhead =" $2 | cut -d" " -f 1)
	NCORES=$(grep "Running" $2 | cut -d" " -f 6)
	FIRST=1
    fi

    if [ $1 == "libGOMP" ];
    then
	LIBGOMP_PERFS=$(grep "overhead =" $2 | cut -d" " -f 4)
    fi

    if [ $1 == "libKOMP" ];
    then
	LIBKOMP_PERFS=$(grep "overhead =" $2 | cut -d" " -f 4)
    fi

    if [ $1 == "intel" ];
    then
	INTEL_PERFS=$(grep "overhead =" $2 | cut -d" " -f 4)
    fi

}

function run_epcc {
    LAUNCHER=
    
    if [ $1 == "libKOMP" ];
    then
	LAUNCHER=kaapi-gomp-wrapper
    fi

    $LAUNCHER $SYNCBENCH $2 > $1.epcclog

    parse_log_file $1 $1.epcclog
}

if [ $# -ne 0 ];
then
    if [ $1 != "--compare" ];
    then
	echo "Bad argument: $1."
	echo "Usage: ./run-epcc.sh epcc-bench nb_runs [--compare <libGOMP libKOMP intel>]"
	exit -1
    else
	shift
	for runtime in $@;
	do
	    run_epcc $runtime $REPS
	done
    fi

    RUNTIMES_TAB=( $@ )

    echo -e "48 cores\t${RUNTIMES_TAB[0]}\t\t${RUNTIMES_TAB[1]}\t${RUNTIMES_TAB[2]}"

    KEYWORDS_TAB=( $KEYWORDS )
    LIBGOMP_TAB=( $LIBGOMP_PERFS )
    LIBKOMP_TAB=( $LIBKOMP_PERFS )

    for keyword in $(seq 0 $(expr $(echo $KEYWORDS | wc -w) - 1));
    do
	if [ $(echo "${LIBGOMP_TAB[$keyword]} > ${LIBKOMP_TAB[$keyword]}" | bc) -eq 1 ];
	then
	    echo -e "${KEYWORDS_TAB[$keyword]}\t\t\033[31m${LIBGOMP_TAB[$keyword]}\t\033[32m${LIBKOMP_TAB[$keyword]}\033[30m"
	else
	    echo -e "${KEYWORDS_TAB[$keyword]}\t\t\033[32m${LIBGOMP_TAB[$keyword]}\t\033[31m${LIBKOMP_TAB[$keyword]}\033[30m"
	fi
    done
fi