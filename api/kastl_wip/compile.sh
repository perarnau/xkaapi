#!/usr/bin/env sh

#XKAAPI_INSTALL_DIR=`pwd`/../install
XKAAPI_INSTALL_DIR=$HOME/install
TBB_INSTALL_DIR=$HOME/install

echo WARNING, check O2
rm main
#g++ -DKAAPI_CONCURRENT_WS=1 -DKASTL_DEBUG=0 -O3 -ggdb -std=gnu++0x -Wall -I$XKAAPI_INSTALL_DIR/include -I. -o main main.cc -L$XKAAPI_INSTALL_DIR/lib -lxkaapi
#g++ -DKAAPI_CONCURRENT_WS=1 -DKASTL_DEBUG=1 -O3 -ggdb -std=gnu++0x -Wall -I$XKAAPI_INSTALL_DIR/include -I. -o main main.cc -L$XKAAPI_INSTALL_DIR/lib -lxkaapi
#g++ -DKASTL_DEBUG=0 -O3 -std=gnu++0x -Wall -I$XKAAPI_INSTALL_DIR/include -I. -o main main.cc -L$XKAAPI_INSTALL_DIR/lib -lxkaapi
#g++ -DKASTL_DEBUG=0 -std=gnu++0x -Wall -O3 -I$XKAAPI_INSTALL_DIR/include -I. -o main main.cc -L$XKAAPI_INSTALL_DIR/lib -lxkaapi -ltbb
/usr/bin/g++-4.5 -DKASTL_DEBUG=0 -std=gnu++0x -Wall -O3 -I$XKAAPI_INSTALL_DIR/include -I$TBB_INSTALL_DIR/include -I. -o main main.cc kastl_workqueue.cpp -L$TBB_INSTALL_DIR/lib -L$XKAAPI_INSTALL_DIR/lib -lxkaapi -lpthread -ltbb
