#!/usr/bin/env bash

CFLAGS='';
HOSTNAME=`hostname`
[ "$HOSTNAME" == "idfreeze" ] && CFLAGS="$CFLAGS -DCONFIG_IDFREEZE=1";
[ "$HOSTNAME" == "idkoiff" ] && CFLAGS="$CFLAGS -DCONFIG_IDKOIFF=1";

gcc $CFLAGS -DCONFIG_STATIC=0 -DCONFIG_OMP=0 -DCONFIG_KAAPI=1 -O3 -march=native -Wall -I$HOME/install/xkaapi_hws/include -o a.out.kaapi foreach.c -L$HOME/install/xkaapi_hws/lib -lkaapi -lm
gcc $CFLAGS -DCONFIG_STATIC=0 -DCONFIG_OMP=1 -DCONFIG_KAAPI=0 -fopenmp -march=native -O3 -Wall -I$HOME/install/xkaapi_hws/include -o a.out.omp foreach.c -lnuma -lm

# static version
# KAAPI_DIR=$HOME/repo/cpb/src
# CFLAGS="$CFLAGS -O3 -Wall -pthread"
# CFLAGS="$CFLAGS -DCONFIG_STATIC=1 -DCONFIG_OMP=0 -DCONFIG_KAAPI=0"
# CFLAGS="$CFLAGS -I$KAAPI_DIR"
# LFLAGS="-lnuma -lpthread -lm"

# SOURCES=""
# SOURCES="$SOURCES main.c"
# SOURCES="$SOURCES $KAAPI_DIR/kaapi_ctor.c"
# SOURCES="$SOURCES $KAAPI_DIR/kaapi_mt.c"
# SOURCES="$SOURCES $KAAPI_DIR/kaapi_proc.c"
# SOURCES="$SOURCES $KAAPI_DIR/kaapi_numa.c"
# SOURCES="$SOURCES $KAAPI_DIR/kaapi_perf.c"

# gcc $CFLAGS -o a.out.static $SOURCES $LFLAGS ;
