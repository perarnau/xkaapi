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

if  KAAPI_API_KAAPIXX
check_PROGRAMS=             \
	test_spawn                \
  test_pointer              \
  test_multipleleave        \
  test_stackpointer         \
  test_nqueens_stackpointer\
  fibo_kaapi++\
  fibo_kaapi++_autopointer \
  fibo_kaapi++_autovar \
  fibo_kaapi++_alloca \
  fibo_kaapi++_opt \
  fibo_kaapi++_opt_thread \
  fibo_kaapi++_opt_sync \
  fibo_kaapi++_sync \
  fibo_kaapi++_cumul \
  fibo_kaapi++_cumul_opt \
  fibo_kaapi++_cumul_userreduc\
  test_perfcounter1 \
  test_spawn_failed 
endif


TESTS=

test_spawn_SOURCES=test_spawn.cpp test_main.cpp test_main.h test_task.h
test_spawn_failed_SOURCES=test_spawn_failed.cpp test_main.cpp test_main.h test_task.h
test_pointer_SOURCES=test_pointer.cpp test_main.cpp test_main.h test_task.h
test_stackpointer_SOURCES=test_stackpointer.cpp test_main.cpp test_main.h test_task.h
test_nqueens_stackpointer_SOURCES=test_nqueens_stackpointer.cpp test_main.cpp test_main.h
test_multipleleave_SOURCES=test_multipleleave.cpp
test_perfcounter1_SOURCES=test_perfcounter1.cpp test_main.cpp

fibo_kaapi___SOURCES=    fibo_kaapi++.cpp test_main.cpp test_main.h fibo_main.cpp
fibo_kaapi___autopointer_SOURCES= fibo_kaapi++_autopointer.cpp test_main.cpp test_main.h fibo_main.cpp
fibo_kaapi___autovar_SOURCES=    fibo_kaapi++_autovar.cpp test_main.cpp test_main.h fibo_main.cpp
fibo_kaapi___alloca_SOURCES=fibo_kaapi++_alloca.cpp test_main.cpp test_main.h fibo_main.cpp
fibo_kaapi___opt_SOURCES=fibo_kaapi++_opt.cpp test_main.cpp test_main.h fibo_main.cpp
fibo_kaapi___opt_thread_SOURCES=fibo_kaapi++_opt_thread.cpp test_main.cpp test_main.h fibo_main.cpp
fibo_kaapi___opt_sync_SOURCES=fibo_kaapi++_opt_sync.cpp test_main.cpp test_main.h fibo_main.cpp
fibo_kaapi___sync_SOURCES=fibo_kaapi++_sync.cpp test_main.cpp test_main.h fibo_main.cpp
fibo_kaapi___cumul_SOURCES=fibo_kaapi++_cumul.cpp test_main.cpp test_main.h fibo_main_cumul.cpp
fibo_kaapi___cumul_opt_SOURCES=fibo_kaapi++_cumul_opt.cpp test_main.cpp test_main.h fibo_main_cumul.cpp
fibo_kaapi___cumul_userreduc_SOURCES=fibo_kaapi++_cumul_userreduc.cpp test_main.cpp test_main.h


test_spawn_CPPFLAGS=$(AM_CPPFLAGS)  $(KAAPIXXPROGS_BUILD_CPPFLAGS)  
test_spawn_CXXFLAGS=$(AM_CXXFLAGS)  $(KAAPIXXPROGS_BUILD_CXXFLAGS)
test_spawn_LDFLAGS =$(AM_LDFLAGS)   $(KAAPIXXPROGS_BUILD_LDFLAGS)
test_spawn_LDADD   =$(LIBS)         $(KAAPIXXPROGS_BUILD_LIBS)

test_spawn_failed_CPPFLAGS=$(AM_CPPFLAGS)  $(KAAPIXXPROGS_BUILD_CPPFLAGS)  
test_spawn_failed_CXXFLAGS=$(AM_CXXFLAGS)  $(KAAPIXXPROGS_BUILD_CXXFLAGS)
test_spawn_failed_LDFLAGS =$(AM_LDFLAGS)   $(KAAPIXXPROGS_BUILD_LDFLAGS)
test_spawn_failed_LDADD   =$(LIBS)         $(KAAPIXXPROGS_BUILD_LIBS)

test_pointer_CPPFLAGS=$(AM_CPPFLAGS)  $(KAAPIXXPROGS_BUILD_CPPFLAGS)  
test_pointer_CXXFLAGS=$(AM_CXXFLAGS)  $(KAAPIXXPROGS_BUILD_CXXFLAGS)
test_pointer_LDFLAGS =$(AM_LDFLAGS)   $(KAAPIXXPROGS_BUILD_LDFLAGS)
test_pointer_LDADD   =$(LIBS)         $(KAAPIXXPROGS_BUILD_LIBS)

test_stackpointer_CPPFLAGS=$(AM_CPPFLAGS)  $(KAAPIXXPROGS_BUILD_CPPFLAGS)  
test_stackpointer_CXXFLAGS=$(AM_CXXFLAGS)  $(KAAPIXXPROGS_BUILD_CXXFLAGS)
test_stackpointer_LDFLAGS =$(AM_LDFLAGS)   $(KAAPIXXPROGS_BUILD_LDFLAGS)
test_stackpointer_LDADD   =$(LIBS)         $(KAAPIXXPROGS_BUILD_LIBS)

test_nqueens_stackpointer_CPPFLAGS=$(AM_CPPFLAGS)  $(KAAPIXXPROGS_BUILD_CPPFLAGS)  
test_nqueens_stackpointer_CXXFLAGS=$(AM_CXXFLAGS)  $(KAAPIXXPROGS_BUILD_CXXFLAGS)
test_nqueens_stackpointer_LDFLAGS =$(AM_LDFLAGS)   $(KAAPIXXPROGS_BUILD_LDFLAGS)
test_nqueens_stackpointer_LDADD   =$(LIBS)         $(KAAPIXXPROGS_BUILD_LIBS)

test_multipleleave_CPPFLAGS=$(AM_CPPFLAGS)  $(KAAPIXXPROGS_BUILD_CPPFLAGS)  
test_multipleleave_CXXFLAGS=$(AM_CXXFLAGS)  $(KAAPIXXPROGS_BUILD_CXXFLAGS)
test_multipleleave_LDFLAGS =$(AM_LDFLAGS)   $(KAAPIXXPROGS_BUILD_LDFLAGS)
test_multipleleave_LDADD   =$(LIBS)         $(KAAPIXXPROGS_BUILD_LIBS)

test_perfcounter1_CPPFLAGS=$(AM_CPPFLAGS)  $(KAAPIXXPROGS_BUILD_CPPFLAGS)  
test_perfcounter1_CXXFLAGS=$(AM_CXXFLAGS)  $(KAAPIXXPROGS_BUILD_CXXFLAGS)
test_perfcounter1_LDFLAGS =$(AM_LDFLAGS)   $(KAAPIXXPROGS_BUILD_LDFLAGS)
test_perfcounter1_LDADD   =$(LIBS)         $(KAAPIXXPROGS_BUILD_LIBS)

fibo_kaapi___CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
fibo_kaapi___CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
fibo_kaapi___LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
fibo_kaapi___LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

fibo_kaapi___autopointer_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
fibo_kaapi___autopointer_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
fibo_kaapi___autopointer_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
fibo_kaapi___autopointer_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

fibo_kaapi___autovar_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
fibo_kaapi___autovar_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
fibo_kaapi___autovar_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
fibo_kaapi___autovar_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

fibo_kaapi___alloca_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
fibo_kaapi___alloca_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
fibo_kaapi___alloca_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
fibo_kaapi___alloca_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

fibo_kaapi___opt_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
fibo_kaapi___opt_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
fibo_kaapi___opt_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
fibo_kaapi___opt_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

fibo_kaapi___opt_thread_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
fibo_kaapi___opt_thread_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
fibo_kaapi___opt_thread_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
fibo_kaapi___opt_thread_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

fibo_kaapi___sync_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
fibo_kaapi___sync_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
fibo_kaapi___sync_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
fibo_kaapi___sync_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

fibo_kaapi___opt_sync_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
fibo_kaapi___opt_sync_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
fibo_kaapi___opt_sync_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
fibo_kaapi___opt_sync_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

fibo_kaapi___cumul_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
fibo_kaapi___cumul_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
fibo_kaapi___cumul_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
fibo_kaapi___cumul_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

fibo_kaapi___cumul_opt_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
fibo_kaapi___cumul_opt_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
fibo_kaapi___cumul_opt_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
fibo_kaapi___cumul_opt_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

fibo_kaapi___cumul_userreduc_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXPROGS_BUILD_CPPFLAGS)
fibo_kaapi___cumul_userreduc_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXPROGS_BUILD_CXXFLAGS)
fibo_kaapi___cumul_userreduc_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIXXPROGS_BUILD_LDFLAGS)
fibo_kaapi___cumul_userreduc_LDADD   =$(LIBS)        $(KAAPIXXPROGS_BUILD_LIBS)

