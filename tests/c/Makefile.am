##
##  xkaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009,2010,2011,2012 INRIA. All rights reserved.
##

include ./Makefile.examples.am

AM_CPPFLAGS=@AM_CPPFLAGS@
AM_CFLAGS=@AM_CFLAGS@ $(STD_C99_FLAGS)
AM_CXXFLAGS=@AM_CXXFLAGS@ $(STD_CXX0X_FLAGS)

if  KAAPI_API_KAAPIC
check_PROGRAMS=   \
	test_begend_parallel \
  test_spawn1 \
  test_spawn2 \
  test_spawn3 \
  test_nqueens_stackpointer\
  fibo_sync \
  fibo_sum
endif


TESTS=

test_begend_parallel_SOURCES=test_begend_parallel.c
test_nqueens_stackpointer_SOURCES=test_nqueens_stackpointer.c
test_spawn1_SOURCES=test_spawn1.c
test_spawn2_SOURCES=test_spawn2.c
test_spawn3_SOURCES=test_spawn3.c
fibo_sync_SOURCES=fibo_sync.c
fibo_sum_SOURCES=fibo_sum.c


test_begend_parallel_CPPFLAGS=$(AM_CPPFLAGS)  $(KAAPICPROGS_BUILD_CPPFLAGS)  
test_begend_parallel_CFLAGS  =$(AM_CFLAGS)    $(KAAPICPROGS_BUILD_CXXFLAGS)
test_begend_parallel_LDFLAGS =$(AM_LDFLAGS)   $(KAAPICPROGS_BUILD_LDFLAGS)
test_begend_parallel_LDADD   =$(LIBS)         $(KAAPICPROGS_BUILD_LIBS)

test_spawn1_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPICPROGS_BUILD_CPPFLAGS)
test_spawn1_CFLAGS  =$(AM_CFLAGS)   $(KAAPICPROGS_BUILD_CFLAGS)
test_spawn1_LDFLAGS =$(AM_LDFLAGS)  $(KAAPICPROGS_BUILD_LDFLAGS)
test_spawn1_LDADD   =$(LIBS)        $(KAAPICPROGS_BUILD_LIBS) -lm

test_spawn2_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPICPROGS_BUILD_CPPFLAGS)
test_spawn2_CFLAGS  =$(AM_CFLAGS)   $(KAAPICPROGS_BUILD_CFLAGS)
test_spawn2_LDFLAGS =$(AM_LDFLAGS)  $(KAAPICPROGS_BUILD_LDFLAGS)
test_spawn2_LDADD   =$(LIBS)        $(KAAPICPROGS_BUILD_LIBS) -lm

test_spawn3_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPICPROGS_BUILD_CPPFLAGS)
test_spawn3_CFLAGS  =$(AM_CFLAGS)   $(KAAPICPROGS_BUILD_CFLAGS)
test_spawn3_LDFLAGS =$(AM_LDFLAGS)  $(KAAPICPROGS_BUILD_LDFLAGS)
test_spawn3_LDADD   =$(LIBS)        $(KAAPICPROGS_BUILD_LIBS) -lm

test_nqueens_stackpointer_CPPFLAGS=$(AM_CPPFLAGS)  $(KAAPICPROGS_BUILD_CPPFLAGS)  
test_nqueens_stackpointer_CFLAGS  =$(AM_CFLAGS)    $(KAAPICPROGS_BUILD_CFLAGS)
test_nqueens_stackpointer_LDFLAGS =$(AM_LDFLAGS)   $(KAAPICPROGS_BUILD_LDFLAGS)
test_nqueens_stackpointer_LDADD   =$(LIBS)         $(KAAPICPROGS_BUILD_LIBS)

fibo_sync_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPICPROGS_BUILD_CPPFLAGS)
fibo_sync_CFLAGS  =$(AM_CFLAGS)   $(KAAPICPROGS_BUILD_CFLAGS)
fibo_sync_LDFLAGS =$(AM_LDFLAGS)  $(KAAPICPROGS_BUILD_LDFLAGS)
fibo_sync_LDADD   =$(LIBS)        $(KAAPICPROGS_BUILD_LIBS) -lm

fibo_sum_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPICPROGS_BUILD_CPPFLAGS)
fibo_sum_CFLAGS  =$(AM_CFLAGS)   $(KAAPICPROGS_BUILD_CFLAGS)
fibo_sum_LDFLAGS =$(AM_LDFLAGS)  $(KAAPICPROGS_BUILD_LDFLAGS)
fibo_sum_LDADD   =$(LIBS)        $(KAAPICPROGS_BUILD_LIBS) -lm
