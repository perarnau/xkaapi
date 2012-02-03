#!/usr/bin/env sh

gfortran -Wall -O3 \
-I$HOME/install/xkaapi_master/include \
main.f \
-L$HOME/install/xkaapi_master/lib -lkaapi -lkaapif -lkaapic -lm
