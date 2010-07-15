#!/usr/bin/env sh
sudo LD_LIBRARY_PATH=/home/lementec/install/lib KAAPI_GPUSET='0~1' KAAPI_CPUSET=0 ./a.out
