#!/usr/bin/env sh
sudo LD_LIBRARY_PATH=/home/lementec/install/xkaapi_gpu/lib KAAPI_GPUSET='0~1' KAAPI_CPUSET=0 ./a.out
