#! /bin/bash

KAAPI_INSTALL_PATH=/Users/broq/soft/install

TESTS=hello

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
    ./$prog
done
