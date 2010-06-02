#!/usr/bin/env sh

ALGOS=''
#ALGOS="$ALGOS find"
#ALGOS="$ALGOS find_if"
#ALGOS="$ALGOS find_first_of"
#ALGOS="$ALGOS accumulate"
#ALGOS="$ALGOS inner_product"
#ALGOS="$ALGOS for_each"
#ALGOS="$ALGOS count"
#ALGOS="$ALGOS count_if"
ALGOS="$ALGOS transform"
#ALGOS="$ALGOS search"
#ALGOS="$ALGOS min_element"
#ALGOS="$ALGOS max_element"
#ALGOS="$ALGOS copy"

for A in $ALGOS; do
    # for L in stl pastl kastl tbb; do
    for L in kastl tbb; do
	make DO=bench LIB=$L ALGO=$A ;
    done
    make DO=check LIB=kastl ALGO=$A ;
#    make DO=check LIB=tbb ALGO=$A ;
done
