#!/usr/bin/env sh

#for A in for_each count transform search min_element max_element; do
for A in accumulate; do
    for L in stl pastl kastl tbb; do
	make DO=bench LIB=$L ALGO=$A ;
    done
    make DO=check LIB=kastl ALGO=$A ;
done
