#!/usr/bin/env sh

#LD_LIBRARY_PATH=$HOME/install/xkaapi_release/lib \
#    KAAPI_CPUSET=`seq -s, 0 15` \
#    ./main_conc_nored

LD_LIBRARY_PATH=$HOME/install/xkaapi_release/lib \
    KAAPI_CPUSET='0:3' \
    ./main_conc_red

#LD_LIBRARY_PATH=$HOME/install/xkaapi_release/lib \
#    KAAPI_CPUSET=`seq -s, 0 15` \
#    ./main_coop_nored
