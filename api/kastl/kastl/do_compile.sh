#!/usr/bin/env sh
g++ -Wall -O3 -std=gnu++0x -DCONFIG_KASTL_DEBUG=0 -I. -I.. -I../.. -I$HOME/install/include -L$HOME/install/lib main.cpp -lkaapi
