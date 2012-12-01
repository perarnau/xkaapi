##
##  xkaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009,2010,2011,2012 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ $(KAAPICPROGS_BUILD_CPPFLAGS)
AM_CFLAGS=@AM_CFLAGS@ $(STD_C99_FLAGS) $(KAAPICPROGS_BUILD_CFLAGS)
AM_CXXFLAGS=@AM_CXXFLAGS@ $(STD_CXX0X_FLAGS) $(KAAPICPROGS_BUILD_CXXFLAGS)
AM_LDFLAGS=@AM_LDFLAGS@ $(KAAPICPROGS_BUILD_LDFLAGS)
AM_LIBS=@AM_LIBS@ $(KAAPICPROGS_BUILD_LIBS)

if  KAAPI_API_KAAPIC
noinst_PROGRAMS=   \
  test_begend_parallel \
  test_spawn1 \
  test_spawn2 \
  test_spawn3 \
  test_nqueens_stackpointer\
  fibo_sync \
  fibo_sum \
  fibo_cumul
endif


TESTS=

test_begend_parallel_SOURCES=test_begend_parallel.c
test_nqueens_stackpointer_SOURCES=test_nqueens_stackpointer.c
test_spawn1_SOURCES=test_spawn1.c
test_spawn2_SOURCES=test_spawn2.c
test_spawn3_SOURCES=test_spawn3.c
fibo_sync_SOURCES=fibo_sync.c
fibo_sum_SOURCES=fibo_sum.c
fibo_cumul_SOURCES=fibo_cumul.c

