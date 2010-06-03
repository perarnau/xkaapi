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
#ALGOS="$ALGOS copy"
#ALGOS="$ALGOS search"
#ALGOS="$ALGOS min_element"
#ALGOS="$ALGOS max_element"
#ALGOS="$ALGOS fill"
#ALGOS="$ALGOS generate"
#ALGOS="$ALGOS inner_product"
#ALGOS="$ALGOS replace"
#ALGOS="$ALGOS replace_if"
#ALGOS="$ALGOS equal"
#ALGOS="$ALGOS mismatch"
#ALGOS="$ALGOS search"
ALGOS="$ALGOS adjacent_find"

for A in $ALGOS; do
    # for L in stl pastl kastl tbb; do
    for L in kastl tbb; do
	make DO=bench LIB=$L ALGO=$A ;
    done
    make DO=check LIB=kastl ALGO=$A ;
#    make DO=check LIB=tbb ALGO=$A ;
done
