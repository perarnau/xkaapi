##
##  xkaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009 INRIA. All rights reserved.
##

include ./Makefile.examples.am

AM_CPPFLAGS=@AM_CPPFLAGS@
AM_CFLAGS=@AM_CFLAGS@ $(STD_C99_FLAGS)
AM_CXXFLAGS=@AM_CXXFLAGS@ $(STD_CXX0X_FLAGS)

check_PROGRAMS= \
	fiboseq                \
	fibo_xkaapi            \
	fibo_atha              \
	fibo_kaapixx           \
	fibo_kaapixx_opt       \
	qsort_cilk2kaapi       \
  matrix_multiply_cilk2kaapi \
	branchbound_apiatha    \
	dgemm_cblas            \
	dgemm_kaapi            \
	dgemm_kaapi_flat       \
	foreach_std            \
	nqueens_apiatha        \
	nqueens_apikaapi       \
	poisson3d-kaapi        \
	poisson3d-xkaapinew    \
  jacobi1D               \
	transform_adapt	       \
	transform_adapt_preempt\
	transform_std

# To correct:
#check_PROGRAMS+=\
#	fibo_xkaapi_adapt
#	transform_concurrent	count_concurrent 	marc_problem	accumulate_while_kastl	foreach_kastl  	transform_kastl  


TESTS=

all-fibo: fiboseq fibo_xkaapi fibo_atha fibo_kaapixx fibo_kaapixx_opt

fiboseq_SOURCES=fiboseq.c
fibo_xkaapi_SOURCES=fibo_xkaapi.c
#fibo_xkaapi_adapt_SOURCES=fibo_xkaapi_adapt.c
fibo_atha_SOURCES=fibo_atha.cpp
fibo_kaapixx_SOURCES=    fibo_kaapixx.cpp     fibo_kaapixx_main.cpp
fibo_kaapixx_opt_SOURCES=fibo_kaapixx_opt.cpp fibo_kaapixx_main.cpp

# Based on Cilk example
qsort_cilk2kaapi_SOURCES=qsort_cilk2kaapi.cpp
matrix_multiply_cilk2kaapi_SOURCES=matrix_multiply_cilk2kaapi.cpp

dgemm_kaapi_SOURCES=dgemm_kaapi.cpp
dgemm_kaapi_flat_SOURCES=dgemm_kaapi_flat.cpp
dgemm_cblas_SOURCES=dgemm_cblas.cpp
poisson3d_kaapi_SOURCES=poisson3d.h poisson3diter.h poisson3d-kaapi.cpp poisson3d.cpp poisson3diter.cpp
poisson3d_xkaapinew_SOURCES=poisson3d.h poisson3diter.h poisson3d-xkaapinew.cpp poisson3d.cpp poisson3diter.cpp
jacobi1D_SOURCES=jacobi1D.cpp
nqueens_apiatha_SOURCES=nqueens_apiatha.cpp
nqueens_apikaapi_SOURCES=nqueens_apikaapi.cpp
foreach_std_SOURCES=foreach_std.cpp random.cpp random.h
foreach_kastl_SOURCES=foreach_kastl.cpp random.cpp random.h
transform_std_SOURCES=transform_std.cpp random.cpp random.h
transform_kastl_SOURCES=transform_kastl.cpp random.cpp random.h
transform_adapt_SOURCES=transform_adapt.h transform_adapt.cpp random.cpp random.h
transform_adapt_preempt_SOURCES=transform_adapt_preempt.h transform_adapt_preempt.cpp random.cpp random.h
#transform_concurrent_SOURCES=transform_concurrent.h transform_concurrent.cpp random.cpp random.h
accumulate_while_kastl_SOURCES=accumulate_while_kastl.cpp random.cpp random.h
#count_concurrent_SOURCES=count_concurrent.h count_concurrent.cpp random.cpp random.h
#marc_problem_SOURCES=marc_problem.cpp
branchbound_apiatha_SOURCES=branchbound_apiatha.cpp

fiboseq_CPPFLAGS=$(AM_CPPFLAGS) $(XKAAPIPROGS_BUILD_CPPFLAGS) -DFIBOCODE -DMAINCODE
fiboseq_CFLAGS  =$(AM_CFLAGS)   $(XKAAPIPROGS_BUILD_CFLAGS)
fiboseq_LDFLAGS =$(AM_LDFLAGS)  $(XKAAPIPROGS_BUILD_LDFLAGS)
fiboseq_LDADD   =$(LIBS)        $(XKAAPIPROGS_BUILD_LIBS)

fibo_xkaapi_CPPFLAGS=$(AM_CPPFLAGS) $(XKAAPIPROGS_BUILD_CPPFLAGS)
fibo_xkaapi_CFLAGS  =$(AM_CFLAGS)   $(XKAAPIPROGS_BUILD_CFLAGS)   $(NESTED_FUNCTIONS_CFLAGS)
fibo_xkaapi_LDFLAGS =$(AM_LDFLAGS)  $(XKAAPIPROGS_BUILD_LDFLAGS)
fibo_xkaapi_LDADD   =$(LIBS)        $(XKAAPIPROGS_BUILD_LIBS)

#fibo_xkaapi_adapt_CPPFLAGS=$(AM_CPPFLAGS) $(XKAAPIPROGS_BUILD_CPPFLAGS)
#fibo_xkaapi_adapt_CFLAGS  =$(AM_CFLAGS)   $(XKAAPIPROGS_BUILD_CFLAGS)   $(NESTED_FUNCTIONS_CFLAGS)
#fibo_xkaapi_adapt_LDFLAGS =$(AM_LDFLAGS)  $(XKAAPIPROGS_BUILD_LDFLAGS)
#fibo_xkaapi_adapt_LDADD   =$(LIBS)        $(XKAAPIPROGS_BUILD_LIBS)

fibo_atha_CPPFLAGS=$(AM_CPPFLAGS) $(ATHAPROGS_BUILD_CPPFLAGS)
fibo_atha_CXXFLAGS=$(AM_CXXFLAGS) $(ATHAPROGS_BUILD_CXXFLAGS)
fibo_atha_LDFLAGS =$(AM_LDFLAGS)  $(ATHAPROGS_BUILD_LDFLAGS)
fibo_atha_LDADD   =$(LIBS)        $(ATHAPROGS_BUILD_LIBS)

fibo_kaapixx_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
fibo_kaapixx_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
fibo_kaapixx_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
fibo_kaapixx_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

fibo_kaapixx_opt_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
fibo_kaapixx_opt_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
fibo_kaapixx_opt_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
fibo_kaapixx_opt_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

qsort_cilk2kaapi_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
qsort_cilk2kaapi_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
qsort_cilk2kaapi_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
qsort_cilk2kaapi_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

matrix_multiply_cilk2kaapi_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
matrix_multiply_cilk2kaapi_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
matrix_multiply_cilk2kaapi_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
matrix_multiply_cilk2kaapi_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

branchbound_apiatha_CPPFLAGS=$(AM_CPPFLAGS) $(ATHAPROGS_BUILD_CPPFLAGS)
branchbound_apiatha_CXXFLAGS=$(AM_CXXFLAGS) $(ATHAPROGS_BUILD_CXXFLAGS)
branchbound_apiatha_LDFLAGS =$(AM_LDFLAGS)  $(ATHAPROGS_BUILD_LDFLAGS)
branchbound_apiatha_LDADD   =$(LIBS)        $(ATHAPROGS_BUILD_LIBS)

#count_concurrent_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS)
#count_concurrent_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS)
#count_concurrent_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
#count_concurrent_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)

dgemm_cblas_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS) -I/System/Library/Frameworks/vecLib.framework/Versions/A/Headers
dgemm_cblas_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
dgemm_cblas_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
dgemm_cblas_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)     -lblas

dgemm_kaapi_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS) -I/System/Library/Frameworks/vecLib.framework/Versions/A/Headers
dgemm_kaapi_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
dgemm_kaapi_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
dgemm_kaapi_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)     -lblas

dgemm_kaapi_flat_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS) -I/System/Library/Frameworks/vecLib.framework/Versions/A/Headers
dgemm_kaapi_flat_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
dgemm_kaapi_flat_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
dgemm_kaapi_flat_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS) -lblas

foreach_std_CPPFLAGS=$(AM_CPPFLAGS)
foreach_std_CXXFLAGS=$(AM_CXXFLAGS)
foreach_std_LDFLAGS =$(AM_LDFLAGS)
foreach_std_LDADD   =$(LIBS)

nqueens_apiatha_CPPFLAGS=$(AM_CPPFLAGS) $(ATHAPROGS_BUILD_CPPFLAGS)
nqueens_apiatha_CXXFLAGS=$(AM_CXXFLAGS) $(ATHAPROGS_BUILD_CXXFLAGS)
nqueens_apiatha_LDFLAGS =$(AM_LDFLAGS)  $(ATHAPROGS_BUILD_LDFLAGS)
nqueens_apiatha_LDADD   =$(LIBS)        $(ATHAPROGS_BUILD_LIBS)

nqueens_apikaapi_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
nqueens_apikaapi_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
nqueens_apikaapi_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
nqueens_apikaapi_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

#marc_problem_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS)
#marc_problem_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS)
#marc_problem_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
#marc_problem_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)

poisson3d_kaapi_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS) -I/System/Library/Frameworks/vecLib.framework/Versions/A/Headers
poisson3d_kaapi_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
poisson3d_kaapi_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
poisson3d_kaapi_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

poisson3d_xkaapinew_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS) -I/System/Library/Frameworks/vecLib.framework/Versions/A/Headers
poisson3d_xkaapinew_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
poisson3d_xkaapinew_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
poisson3d_xkaapinew_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

jacobi1D_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS) -I/System/Library/Frameworks/vecLib.framework/Versions/A/Headers
jacobi1D_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
jacobi1D_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
jacobi1D_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

accumulate_while_kastl_CPPFLAGS=$(AM_CPPFLAGS) $(KASTLPROGS_BUILD_CPPFLAGS)
accumulate_while_kastl_CXXFLAGS=$(AM_CXXFLAGS) $(KASTLPROGS_BUILD_CXXFLAGS)
accumulate_while_kastl_LDFLAGS =$(AM_LDFLAGS)  $(KASTLPROGS_BUILD_LDFLAGS)
accumulate_while_kastl_LDADD   =$(LIBS)        $(KASTLPROGS_BUILD_LIBS)

foreach_kastl_CPPFLAGS=$(AM_CPPFLAGS) $(KASTLPROGS_BUILD_CPPFLAGS)
foreach_kastl_CXXFLAGS=$(AM_CXXFLAGS) $(KASTLPROGS_BUILD_CXXFLAGS)
foreach_kastl_LDFLAGS =$(AM_LDFLAGS)  $(KASTLPROGS_BUILD_LDFLAGS)
foreach_kastl_LDADD   =$(LIBS)        $(KASTLPROGS_BUILD_LIBS)

transform_adapt_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
transform_adapt_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
transform_adapt_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
transform_adapt_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

transform_adapt_preempt_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
transform_adapt_preempt_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
transform_adapt_preempt_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
transform_adapt_preempt_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

#transform_concurrent_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1PROGS_BUILD_CPPFLAGS)
#transform_concurrent_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1PROGS_BUILD_CXXFLAGS)
#transform_concurrent_LDFLAGS =$(AM_LDFLAGS)  $(ATHA1PROGS_BUILD_LDFLAGS)
#transform_concurrent_LDADD   =$(LIBS)        $(ATHA1PROGS_BUILD_LIBS)

transform_kastl_CPPFLAGS=$(AM_CPPFLAGS) $(KASTLPROGS_BUILD_CPPFLAGS)
transform_kastl_CXXFLAGS=$(AM_CXXFLAGS) $(KASTLPROGS_BUILD_CXXFLAGS)
transform_kastl_LDFLAGS =$(AM_LDFLAGS)  $(KASTLPROGS_BUILD_LDFLAGS)
transform_kastl_LDADD   =$(LIBS)        $(KASTLPROGS_BUILD_LIBS)

transform_std_CPPFLAGS=$(AM_CPPFLAGS)
transform_std_CXXFLAGS=$(AM_CXXFLAGS)
transform_std_LDFLAGS =$(AM_LDFLAGS)
transform_std_LDADD   =$(LIBS)

