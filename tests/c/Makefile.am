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
	test_begend_parallel
endif


TESTS=

test_begend_parallel_SOURCES=test_begend_parallel.c


test_begend_parallel_CPPFLAGS=$(AM_CPPFLAGS)  $(KAAPICPROGS_BUILD_CPPFLAGS)  
test_begend_parallel_CFLAGS  =$(AM_CFLAGS)    $(KAAPICPROGS_BUILD_CXXFLAGS)
test_begend_parallel_LDFLAGS =$(AM_LDFLAGS)   $(KAAPICPROGS_BUILD_LDFLAGS)
test_begend_parallel_LDADD   =$(LIBS)         $(KAAPICPROGS_BUILD_LIBS)
