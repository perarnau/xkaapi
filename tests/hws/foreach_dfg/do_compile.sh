#!/usr/bin/env bash

CFLAGS='';
HOSTNAME=`hostname`
[ "$HOSTNAME" == "idfreeze" ] && CFLAGS="$CFLAGS -DCONFIG_IDFREEZE=1";
[ "$HOSTNAME" == "idkoiff" ] && CFLAGS="$CFLAGS -DCONFIG_IDKOIFF=1";

gcc $CFLAGS -DCONFIG_STATIC=0 -DCONFIG_KAAPI=1 -O3 -march=native -Wall -I$HOME/install/xkaapi_hws/include main.c -L$HOME/install/xkaapi_hws/lib -lkaapi -lm -lnuma
