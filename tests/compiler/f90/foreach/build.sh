#!/bin/bash
#
gfortran -fopenmp main.f90 -o main_omp
gfortran main.f90 -o main_single

LD_LIBRARY_PATH=/usr/lib/jvm/java-6-openjdk/jre/lib/i386/client \
$HOME/install/xkaapi_rose/bin/kaapi_f2f main.f90
gfortran -std=f95 rose_main.f90 -o main_xkaapi
