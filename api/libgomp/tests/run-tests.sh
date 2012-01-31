#! /bin/bash

KAAPI_INSTALL_PATH=

# KAAPI_INSTALL_PATH=$HOME/soft/install/xkaapi

if [ "x$KAAPI_INSTALL_PATH" == "x" ];
then
    echo "Fatal: KAAPI_INSTALL_PATH must be defined in run-tests.sh."
    exit
fi


TESTS="parallel barrier critical single shared"

make clean
make

if [ $# -ne 0 ]; 
then
    TESTS=$@
fi

(uname -a | grep Darwin) > /dev/null 2> /dev/null
DARWIN_SYS=$?

if [ $DARWIN_SYS -eq 0 ]; 
then
    export DYLD_FORCE_FLAT_NAMESPACE=1
    export DYLD_INSERT_LIBRARIES="$KAAPI_INSTALL_PATH/lib/libkaapic.dylib:$KAAPI_INSTALL_PATH/lib/libgomp.dylib"
else
    export LD_PRELOAD="../.libs/libgomp.so"
fi
  
for prog in $TESTS; 
do
    echo "*** Running \"$prog\" test ***"
    time ./$prog
done
