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
	fibo_apikaapi          \
	fibo_apikaapi_opt      \
	dgemm_kaapi            \
	dgemm_kaapi_flat       \
	dgemm_cblas            \
	poisson3d-kaapi        \
	nqueens_apiatha        \
	nqueens_apikaapi       \
  foreach_std            \
  foreach_kastl          \
  transform_std          \
  transform_kastl        \
	transform_adapt	       \
	transform_adapt_preempt\
	accumulate_while_kastl \
	branchbound_apiatha

# To correct:
#	fibo_adapt transform_concurrent	count_concurrent 	marc_problem 

TESTS=

fiboseq_SOURCES=fiboseq.c
fibo_SOURCES=fibo.c
fibo_adapt_SOURCES=fibo_adapt.c
fibo_apiatha_SOURCES=fibo_apiatha.cpp
fibo_apikaapi_SOURCES=fibo_apikaapi.cpp fibo_apikaapi_main.cpp
fibo_apikaapi_opt_SOURCES=fibo_apikaapi_opt.cpp fibo_apikaapi_main.cpp
dgemm_kaapi_SOURCES=dgemm_kaapi.cpp
dgemm_kaapi_flat_SOURCES=dgemm_kaapi_flat.cpp
dgemm_cblas_SOURCES=dgemm_cblas.cpp
poisson3d_kaapi_SOURCES=poisson3d-kaapi.cpp poisson3d.cpp poisson3diter.cpp
nqueens_apiatha_SOURCES=nqueens_apiatha.cpp
nqueens_apikaapi_SOURCES=nqueens_apikaapi.cpp
foreach_std_SOURCES=foreach_std.cpp random.cpp random.h
foreach_kastl_SOURCES=foreach_kastl.cpp random.cpp random.h
transform_std_SOURCES=transform_std.cpp random.cpp random.h
transform_kastl_SOURCES=transform_kastl.cpp random.cpp random.h
transform_adapt_SOURCES=transform_adapt.h transform_adapt.cpp random.cpp random.h
transform_adapt_preempt_SOURCES=transform_adapt_preempt.h transform_adapt_preempt.cpp random.cpp random.h
transform_concurrent_SOURCES=transform_concurrent.h transform_concurrent.cpp random.cpp random.h
accumulate_while_kastl_SOURCES=accumulate_while_kastl.cpp random.cpp random.h
count_concurrent_SOURCES=count_concurrent.h count_concurrent.cpp random.cpp random.h
marc_problem_SOURCES=marc_problem.cpp
branchbound_apiatha_SOURCES=branchbound_apiatha.cpp

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

fibo_apiatha_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api/atha -I$(top_srcdir)/api/kaapixx
fibo_apiatha_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
fibo_apiatha_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
fibo_apiatha_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la

fibo_apikaapi_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api/kaapixx
fibo_apikaapi_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
fibo_apikaapi_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
fibo_apikaapi_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la

fibo_apikaapi_opt_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api/kaapixx
fibo_apikaapi_opt_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
fibo_apikaapi_opt_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
fibo_apikaapi_opt_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la

dgemm_kaapi_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api/atha -I$(top_srcdir)/api/kaapixx -I/System/Library/Frameworks/vecLib.framework/Versions/A/Headers
dgemm_kaapi_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
dgemm_kaapi_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
dgemm_kaapi_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la -lblas

dgemm_kaapi_flat_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api/kaapixx -I/System/Library/Frameworks/vecLib.framework/Versions/A/Headers
dgemm_kaapi_flat_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
dgemm_kaapi_flat_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
dgemm_kaapi_flat_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la -lblas

dgemm_cblas_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api/kaapixx -I/System/Library/Frameworks/vecLib.framework/Versions/A/Headers
dgemm_cblas_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
dgemm_cblas_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
dgemm_cblas_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la -lblas

poisson3d_kaapi_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api/kaapixx -I/System/Library/Frameworks/vecLib.framework/Versions/A/Headers
poisson3d_kaapi_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
poisson3d_kaapi_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
poisson3d_kaapi_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la

nqueens_apiatha_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api/atha -I$(top_srcdir)/api/kaapixx
nqueens_apiatha_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
nqueens_apiatha_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
nqueens_apiatha_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la

nqueens_apikaapi_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api/kaapixx
nqueens_apikaapi_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
nqueens_apikaapi_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
nqueens_apikaapi_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la

branchbound_apiatha_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api/atha -I$(top_srcdir)/api/kaapixx
branchbound_apiatha_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
branchbound_apiatha_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
branchbound_apiatha_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la

foreach_std_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api -I$(top_srcdir)/api/kaapixx
foreach_std_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
foreach_std_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
foreach_std_LDADD   =$(LIBS)

foreach_kastl_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api -I$(top_srcdir)/api/kaapixx
foreach_kastl_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
foreach_kastl_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
foreach_kastl_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la

transform_std_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api -I$(top_srcdir)/api/kaapixx
transform_std_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
transform_std_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
transform_std_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la

transform_kastl_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api -I$(top_srcdir)/api/kaapixx
transform_kastl_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
transform_kastl_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
transform_kastl_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la

transform_adapt_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api -I$(top_srcdir)/api/kaapixx
transform_adapt_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
transform_adapt_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
transform_adapt_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la

transform_adapt_preempt_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api -I$(top_srcdir)/api/kaapixx
transform_adapt_preempt_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
transform_adapt_preempt_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
transform_adapt_preempt_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la

transform_concurrent_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api -I$(top_srcdir)/api/kaapixx
transform_concurrent_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
transform_concurrent_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
transform_concurrent_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la

accumulate_while_kastl_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api -I$(top_srcdir)/api/kaapixx
accumulate_while_kastl_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
accumulate_while_kastl_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
accumulate_while_kastl_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la

count_concurrent_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api -I$(top_srcdir)/api/kaapixx
count_concurrent_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
count_concurrent_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
count_concurrent_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la

marc_problem_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS) -I$(top_srcdir)/api/atha -I$(top_srcdir)/api/kaapixx
marc_problem_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
marc_problem_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
marc_problem_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)     ../api/libxkaapixx.la ../src/libxkaapi.la

