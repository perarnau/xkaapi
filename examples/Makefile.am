##
##  xkaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009 INRIA. All rights reserved.
##

include ./Makefile.examples.am

AM_CPPFLAGS=@AM_CPPFLAGS@
AM_CFLAGS=@AM_CFLAGS@ $(STD_C99_FLAGS)
AM_CXXFLAGS=@AM_CXXFLAGS@


check_PROGRAMS=

include accumulate.am
include cilk.am
include fibo.am
include find.am
include for_each.am
include hello.am
include nqueens.am
include poisson3D.am


check_PROGRAMS+= \
	branchbound_apiatha

TESTS=  

#accumulate_while_kastl_SOURCES=accumulate_while_kastl.cpp random.cpp random.h

branchbound_apiatha_SOURCES=branchbound_apiatha.cpp



branchbound_apiatha_CPPFLAGS=$(AM_CPPFLAGS) $(ATHAPROGS_BUILD_CPPFLAGS)
branchbound_apiatha_CXXFLAGS=$(AM_CXXFLAGS) $(ATHAPROGS_BUILD_CXXFLAGS)
branchbound_apiatha_LDFLAGS =$(AM_LDFLAGS)  $(ATHAPROGS_BUILD_LDFLAGS)
branchbound_apiatha_LDADD   =$(LIBS)        $(ATHAPROGS_BUILD_LIBS)

