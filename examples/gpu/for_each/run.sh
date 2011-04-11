#!/usr/bin/env sh

export LD_LIBRARY_PATH=/usr/local/stow/cuda-4.0/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/install/xkaapi_gpu/lib:$LD_LIBRARY_PATH

export KAAPI_CPUSET=0
export KAAPI_GPUSET=''

case `hostname` in
    "idgraf")
	echo "idgraf" ;
	export KAAPI_GPUSET="$KAAPI_GPUSET,0~6" ;
	export KAAPI_GPUSET="$KAAPI_GPUSET,1~1" ;
	export KAAPI_GPUSET="$KAAPI_GPUSET,3~3" ;
	export KAAPI_GPUSET="$KAAPI_GPUSET,4~4" ;
	;;
    "idkoiff")
	export KAAPI_GPUSET="$KAAPI_GPUSET,0~6" ;
	export KAAPI_GPUSET="$KAAPI_GPUSET,1~1" ;
	;;
esac

# 33554432 = 32 * 1024 * 1024
KAAPI_STACKSIZE=260046848 ./a.out 33554432
