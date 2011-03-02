#!/usr/bin/env sh
LD_LIBRARY_PATH=$HOME/install/xkaapi_master/lib:$HOME/install/lib \
KAAPI_PERF_PAPIES=PAPI_L1_DCM,PAPI_TOT_CYC,PAPI_FP_OPS \
KAAPI_CPUSET=0,1,2,3 \
./a.out
