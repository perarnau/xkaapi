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

check_PROGRAMS= \
  hello_world             \
	fiboseq                 \
	fibo_xkaapi             \
	fibo_atha               \
	fibo_kaapixx            \
	fibo_kaapixx_opt        \
	fibo_kaapixx_sync       \
	qsort_cilk2kaapi        \
	matrix_multiply_cilk2kaapi \
	branchbound_apiatha     \
	nqueens_apiatha         \
	nqueens_apikaapi        \
	poisson3d_kaapi_xx      \
	for_each_0              \
	for_each_1              \
	find_xkaapi		\
  for_each_rec_xx         \
  for_each_0xx            \
  for_each_1xx            \
  for_each_2xx

TESTS=  

adaptive_API=


lambda_TESTS=  \
  for_each_0xx_lambda\
  for_each_1xx_lambda

all-fibo: fiboseq fibo_xkaapi fibo_atha fibo_kaapixx fibo_kaapixx_opt

hello_world_SOURCES=hello/hello_world.cpp
fiboseq_SOURCES=fibo/fiboseq.c
fibo_xkaapi_SOURCES=fibo/fibo_xkaapi.c
fibo_atha_SOURCES=fibo/fibo_atha.cpp
fibo_kaapixx_SOURCES=    fibo/fibo_kaapixx.cpp
fibo_kaapixx_opt_SOURCES=fibo/fibo_kaapixx_opt.cpp
fibo_kaapixx_sync_SOURCES=fibo/fibo_kaapixx_sync.cpp 

# Based on Cilk example
qsort_cilk2kaapi_SOURCES=cilk/qsort_cilk2kaapi.cpp
matrix_multiply_cilk2kaapi_SOURCES=cilk/matrix_multiply_cilk2kaapi.cpp

# Domain decomposition
poisson3d_kaapi_xx_SOURCES=poisson3D/poisson3d.h poisson3D/poisson3diter.h poisson3D/poisson3d-kaapi_xx.cpp poisson3D/poisson3d.cpp poisson3D/poisson3diter.cpp

#Nqueens program.
nqueens_apiatha_SOURCES=nqueens/nqueens_apiatha.cpp
nqueens_apikaapi_SOURCES=nqueens/nqueens_apikaapi.cpp

#For each variation
for_each_0_SOURCES=for_each/for_each_0.c
for_each_1_SOURCES=for_each/for_each_1.c
for_each_rec_xx_SOURCES=for_each/for_each_rec_xx.cpp
for_each_0xx_SOURCES=for_each/for_each_0xx.cpp
for_each_0xx_lambda_SOURCES=for_each/for_each_0xx_lambda.cpp
for_each_1xx_SOURCES=for_each/for_each_1xx.cpp
for_each_2xx_SOURCES=for_each/for_each_2xx.cpp

#foreach_std_SOURCES=foreach_std.cpp random.cpp random.h
#foreach_kastl_SOURCES=foreach_kastl.cpp random.cpp random.h
#accumulate_while_kastl_SOURCES=accumulate_while_kastl.cpp random.cpp random.h

#Find STL algorithm
find_xkaapi_SOURCES=find/find.c

branchbound_apiatha_SOURCES=branchbound_apiatha.cpp

hello_world_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
hello_world_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
hello_world_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
hello_world_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

fiboseq_CPPFLAGS=$(AM_CPPFLAGS) $(XKAAPIPROGS_BUILD_CPPFLAGS) -DFIBOCODE -DMAINCODE
fiboseq_CFLAGS  =$(AM_CFLAGS)   $(XKAAPIPROGS_BUILD_CFLAGS)
fiboseq_LDFLAGS =$(AM_LDFLAGS)  $(XKAAPIPROGS_BUILD_LDFLAGS)
fiboseq_LDADD   =$(LIBS)        $(XKAAPIPROGS_BUILD_LIBS)

fibo_xkaapi_CPPFLAGS=$(AM_CPPFLAGS) $(XKAAPIPROGS_BUILD_CPPFLAGS)
fibo_xkaapi_CFLAGS  =$(AM_CFLAGS)   $(XKAAPIPROGS_BUILD_CFLAGS) $(NESTED_FUNCTIONS_CFLAGS)
fibo_xkaapi_LDFLAGS =$(AM_LDFLAGS)  $(XKAAPIPROGS_BUILD_LDFLAGS)
fibo_xkaapi_LDADD   =$(LIBS)        $(XKAAPIPROGS_BUILD_LIBS)

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

fibo_kaapixx_sync_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
fibo_kaapixx_sync_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
fibo_kaapixx_sync_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
fibo_kaapixx_sync_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

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

nqueens_apiatha_CPPFLAGS=$(AM_CPPFLAGS) $(ATHAPROGS_BUILD_CPPFLAGS)
nqueens_apiatha_CXXFLAGS=$(AM_CXXFLAGS) $(ATHAPROGS_BUILD_CXXFLAGS)
nqueens_apiatha_LDFLAGS =$(AM_LDFLAGS)  $(ATHAPROGS_BUILD_LDFLAGS)
nqueens_apiatha_LDADD   =$(LIBS)        $(ATHAPROGS_BUILD_LIBS)

nqueens_apikaapi_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
nqueens_apikaapi_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
nqueens_apikaapi_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
nqueens_apikaapi_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

poisson3d_kaapi_xx_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
poisson3d_kaapi_xx_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
poisson3d_kaapi_xx_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
poisson3d_kaapi_xx_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

for_each_0_CPPFLAGS=$(AM_CPPFLAGS) $(XKAAPIPROGS_BUILD_CPPFLAGS)
for_each_0_CFLAGS  =$(AM_CFLAGS)   $(XKAAPIPROGS_BUILD_CFLAGS)
for_each_0_LDFLAGS =$(AM_LDFLAGS)  $(XKAAPIPROGS_BUILD_LDFLAGS)
for_each_0_LDADD   =$(LIBS)        $(XKAAPIPROGS_BUILD_LIBS) -lm

for_each_1_CPPFLAGS=$(AM_CPPFLAGS) $(XKAAPIPROGS_BUILD_CPPFLAGS)
for_each_1_CFLAGS  =$(AM_CFLAGS)   $(XKAAPIPROGS_BUILD_CFLAGS)
for_each_1_LDFLAGS =$(AM_LDFLAGS)  $(XKAAPIPROGS_BUILD_LDFLAGS)
for_each_1_LDADD   =$(LIBS)        $(XKAAPIPROGS_BUILD_LIBS) -lm

for_each_rec_xx_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
for_each_rec_xx_CFLAGS  =$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CFLAGS)
for_each_rec_xx_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
for_each_rec_xx_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS) -lm

for_each_0xx_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
for_each_0xx_CFLAGS  =$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CFLAGS)
for_each_0xx_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
for_each_0xx_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS) -lm

for_each_0xx_lambda_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
for_each_0xx_lambda_CFLAGS  =$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CFLAGS)
for_each_0xx_lambda_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
for_each_0xx_lambda_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS) -lm

for_each_1xx_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
for_each_1xx_CFLAGS  =$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CFLAGS)
for_each_1xx_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
for_each_1xx_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS) -lm

for_each_2xx_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
for_each_2xx_CFLAGS  =$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CFLAGS)
for_each_2xx_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
for_each_2xx_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS) -lm

find_xkaapi_CPPFLAGS=$(AM_CPPFLAGS) $(XKAAPIPROGS_BUILD_CPPFLAGS)
find_xkaapi_CFLAGS  =$(AM_CFLAGS)   $(XKAAPIPROGS_BUILD_CFLAGS)
find_xkaapi_LDFLAGS =$(AM_LDFLAGS)  $(XKAAPIPROGS_BUILD_LDFLAGS)
find_xkaapi_LDADD   =$(LIBS)        $(XKAAPIPROGS_BUILD_LIBS) -lm

#accumulate_while_kastl_CPPFLAGS=$(AM_CPPFLAGS) $(KASTLPROGS_BUILD_CPPFLAGS)
#accumulate_while_kastl_CXXFLAGS=$(AM_CXXFLAGS) $(KASTLPROGS_BUILD_CXXFLAGS)
#accumulate_while_kastl_LDFLAGS =$(AM_LDFLAGS)  $(KASTLPROGS_BUILD_LDFLAGS)
#accumulate_while_kastl_LDADD   =$(LIBS)        $(KASTLPROGS_BUILD_LIBS)

#foreach_kastl_CPPFLAGS=$(AM_CPPFLAGS) $(KASTLPROGS_BUILD_CPPFLAGS)
#foreach_kastl_CXXFLAGS=$(AM_CXXFLAGS) $(KASTLPROGS_BUILD_CXXFLAGS)
#foreach_kastl_LDFLAGS =$(AM_LDFLAGS)  $(KASTLPROGS_BUILD_LDFLAGS)
#foreach_kastl_LDADD   =$(LIBS)        $(KASTLPROGS_BUILD_LIBS)

#foreach_std_CPPFLAGS=$(AM_CPPFLAGS)
#foreach_std_CXXFLAGS=$(AM_CXXFLAGS)
#foreach_std_LDFLAGS =$(AM_LDFLAGS)
#foreach_std_LDADD   =$(LIBS)
