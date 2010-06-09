#!/usr/bin/env sh

ALGOS=''
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
LIBS="$LIBS pastl"
LIBS="$LIBS pthread"

DOS=''
DOS="$DOS bench"
#DOS="$DOS check"

for D in $DOS; do
    for A in $ALGOS; do
	for L in $LIBS; do
	    make DO=$D LIB=$L ALGO=$A ;
	done
    done
done
