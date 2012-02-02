#! /bin/bash

if [ $# -eq 0 ]; 
then
    echo "Usage: ./run-single-test.sh testname [testargs]"
    exit
fi

#KAAPI_INSTALL_PATH=

# KAAPI_INSTALL_PATH=$HOME/soft/install/xkaapi

if [ "x$KAAPI_INSTALL_PATH" == "x" ];
then
    echo "Fatal: KAAPI_INSTALL_PATH must be defined in run-single-test.sh."
    exit
fi

(uname -a | grep Darwin) > /dev/null 2> /dev/null
DARWIN_SYS=$?

if [ $DARWIN_SYS -eq 0 ]; 
then
    export DYLD_FORCE_FLAT_NAMESPACE=1
    export DYLD_INSERT_LIBRARIES="$KAAPI_INSTALL_PATH/lib/libkaapic.dylib:$KAAPI_INSTALL_PATH/lib/libgomp.dylib"
else
    export LD_PRELOAD="$KAAPI_INSTALL_PATH/lib/libkaapic.so:$KAAPI_INSTALL_PATH/lib/libgomp.so"
fi

testname=$1
shift

# Run the test  
time $testname $@
