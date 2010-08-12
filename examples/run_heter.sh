#!/usr/bin/env sh

# 1 cpu, 1 gpu partitions
#sudo LD_LIBRARY_PATH=/home/lementec/install/xkaapi_gpu/lib KAAPI_GPUSET='0~8' KAAPI_CPUSET=0 ./a.out

# 2 cpus, 1 gpu partitions
sudo LD_LIBRARY_PATH=/home/lementec/install/xkaapi_gpu/lib KAAPI_GPUSET='0~8' KAAPI_CPUSET=0,1 ./a.out
