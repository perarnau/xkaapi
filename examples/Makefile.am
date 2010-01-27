##
##  xkaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009 INRIA. All rights reserved.
##

include ./Makefile.examples.am

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src

check_PROGRAMS= \
	fiboseq                \
	fibo                   \
	fibo_apiatha           \
	fibo_apiatha2          \
	fibo_apiatha3          \
	nqueens_apiatha        \
	branchbound_apiatha    \
	transform_adapt	       \
	transform_concurrent   \
	count_concurrent       \
	marc_problem          

# To correct:
#	fibo_adapt             

TESTS=

fiboseq_SOURCES=fiboseq.c
fibo_SOURCES=fibo.c
fibo_adapt_SOURCES=fibo_adapt.c
fibo_apiatha_SOURCES=fibo_apiatha.cpp
fibo_apiatha2_SOURCES=fibo_apiatha2.cpp
fibo_apiatha3_SOURCES=fibo_apiatha3.cpp
nqueens_apiatha_SOURCES=nqueens_apiatha.cpp
branchbound_apiatha_SOURCES=branchbound_apiatha.cpp
transform_adapt_SOURCES=transform_adapt.h transform_adapt.cpp random.cpp random.h
transform_concurrent_SOURCES=transform_concurrent.h transform_concurrent.cpp random.cpp random.h
count_concurrent_SOURCES=count_concurrent.h count_concurrent.cpp random.cpp random.h
marc_problem_SOURCES=marc_problem.cpp

fiboseq_CPPFLAGS=$(AM_CPPFLAGS) $(XKAAPIPROGS_BUILD_CPPFLAGS) -DFIBOCODE -DMAINCODE
fiboseq_CFLAGS  =$(AM_CFLAGS)   $(XKAAPIPROGS_BUILD_CFLAGS)  $(STD_C99_FLAGS) 
fiboseq_LDFLAGS =$(AM_LDFLAGS)  $(XKAAPIPROGS_BUILD_LDFLAGS)
fiboseq_LDADD   =$(LIBS)        $(XKAAPIPROGS_BUILD_LIBS)     ../src/libxkaapi.la

fibo_CPPFLAGS=$(AM_CPPFLAGS) $(XKAAPIPROGS_BUILD_CPPFLAGS)
fibo_CFLAGS  =$(AM_CFLAGS)   $(XKAAPIPROGS_BUILD_CFLAGS)   $(STD_C99_FLAGS) $(NESTED_FUNCTIONS_CFLAGS)
fibo_LDFLAGS =$(AM_LDFLAGS)  $(XKAAPIPROGS_BUILD_LDFLAGS)
fibo_LDADD   =$(LIBS)        $(XKAAPIPROGS_BUILD_LIBS)     ../src/libxkaapi.la

fibo_adapt_CPPFLAGS=$(AM_CPPFLAGS) $(XKAAPIPROGS_BUILD_CPPFLAGS)
fibo_adapt_CFLAGS  =$(AM_CFLAGS)   $(XKAAPIPROGS_BUILD_CFLAGS)   $(STD_C99_FLAGS) $(NESTED_FUNCTIONS_CFLAGS)
fibo_adapt_LDFLAGS =$(AM_LDFLAGS)  $(XKAAPIPROGS_BUILD_LDFLAGS)
fibo_adapt_LDADD   =$(LIBS)        $(XKAAPIPROGS_BUILD_LIBS)     ../src/libxkaapi.la

fibo_apiatha_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api-atha
fibo_apiatha_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
fibo_apiatha_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
fibo_apiatha_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api-atha/libatha1.la ../src/libxkaapi.la

fibo_apiatha2_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api-atha
fibo_apiatha2_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
fibo_apiatha2_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
fibo_apiatha2_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api-atha/libatha1.la ../src/libxkaapi.la

fibo_apiatha3_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api-atha
fibo_apiatha3_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
fibo_apiatha3_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
fibo_apiatha3_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api-atha/libatha1.la ../src/libxkaapi.la

nqueens_apiatha_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api-atha
nqueens_apiatha_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
nqueens_apiatha_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
nqueens_apiatha_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api-atha/libatha1.la ../src/libxkaapi.la

branchbound_apiatha_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api-atha
branchbound_apiatha_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
branchbound_apiatha_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
branchbound_apiatha_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api-atha/libatha1.la ../src/libxkaapi.la

transform_adapt_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api-atha
transform_adapt_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
transform_adapt_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
transform_adapt_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api-atha/libatha1.la ../src/libxkaapi.la

transform_concurrent_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api-atha
transform_concurrent_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
transform_concurrent_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
transform_concurrent_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api-atha/libatha1.la ../src/libxkaapi.la

count_concurrent_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api-atha
count_concurrent_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
count_concurrent_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
count_concurrent_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api-atha/libatha1.la ../src/libxkaapi.la

marc_problem_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api-atha
marc_problem_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
marc_problem_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
marc_problem_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api-atha/libatha1.la ../src/libxkaapi.la

