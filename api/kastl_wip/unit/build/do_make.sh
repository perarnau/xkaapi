#!/usr/bin/env sh

#for A in for_each count transform search min_element max_element; do
for A in inner_product; do
    for L in stl pastl kastl tbb; do
#    for L in tbb; do
	make DO=bench LIB=$L ALGO=$A ;
    done
    make DO=check LIB=kastl ALGO=$A ;
    make DO=check LIB=tbb ALGO=$A ;
done
