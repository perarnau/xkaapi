#!/usr/bin/env sh

#XKAAPI_INSTALL_DIR=`pwd`/../install
XKAAPI_INSTALL_DIR=/home/lementec/install

echo WARNING, check O2
rm main
#g++ -DKAAPI_CONCURRENT_WS=1 -DKASTL_DEBUG=0 -O3 -ggdb -std=gnu++0x -Wall -I$XKAAPI_INSTALL_DIR/include -I. -o main main.cc -L$XKAAPI_INSTALL_DIR/lib -lkaapi
#g++ -DKAAPI_CONCURRENT_WS=1 -DKASTL_DEBUG=1 -O3 -ggdb -std=gnu++0x -Wall -I$XKAAPI_INSTALL_DIR/include -I. -o main main.cc -L$XKAAPI_INSTALL_DIR/lib -lkaapi
g++ -DKASTL_DEBUG=0 -O3 -std=gnu++0x -Wall -I$XKAAPI_INSTALL_DIR/include -I. -o main main.cc -L$XKAAPI_INSTALL_DIR/lib -lkaapi
