#! /bin/bash

TESTS=hello

make clean
make

if [ $# -ne 0 ]; 
then
    TESTS=$@
fi
  
for prog in $TESTS; 
do
    echo "*** Running \"$prog\" test ***"
    LD_PRELOAD=../.libs/libgomp.so ./$prog
done
