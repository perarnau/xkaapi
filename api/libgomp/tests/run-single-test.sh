#! /bin/bash

if [ $# -ne 1 ]; 
then
    echo "Usage: ./run-single-test.sh testname"
fi
  
LD_PRELOAD=../.libs/libgomp.so ./$1
