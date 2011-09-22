#!/usr/bin/env bash
CFLAGS='';
HOSTNAME=`hostname`
[ "$HOSTNAME" == "idfreeze" ] && CFLAGS="$CFLAGS -DCONFIG_IDFREEZE=1";
[ "$HOSTNAME" == "idkoiff" ] && CFLAGS="$CFLAGS -DCONFIG_IDKOIFF=1";
gcc $CFLAGS -DCONFIG_OMP=0 -DCONFIG_KAAPI=1 -O3 -Wall -I$HOME/install/xkaapi_hws/include -o a.out.kaapi tests/hws/foreach.c -L$HOME/install/xkaapi_hws/lib -lkaapi -lm
gcc $CFLAGS -DCONFIG_OMP=1 -DCONFIG_KAAPI=0 -fopenmp -O3 -Wall -I$HOME/install/xkaapi_hws/include -o a.out.omp tests/hws/foreach.c -lnuma -lm
