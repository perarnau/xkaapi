#!/usr/bin/env sh

for A in for_each count transform search; do
    for L in stl pastl kastl tbb; do
	make DO=bench LIB=$L ALGO=$A ;
    done
    make DO=check LIB=kastl ALGO=$A ;
done
