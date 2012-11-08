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
check_PROGRAMS=   \
	test_spawn      \
  test_pointer
endif


TESTS=

test_spawn_SOURCES=test_spawn.cpp test_main.cpp test_main.h test_task.h
test_pointer_SOURCES=test_pointer.cpp test_main.cpp test_main.h test_task.h


test_spawn_CPPFLAGS=$(AM_CPPFLAGS)  $(KAAPIXXPROGS_BUILD_CPPFLAGS)  
test_spawn_CXXFLAGS=$(AM_CXXFLAGS)  $(KAAPIXXPROGS_BUILD_CXXFLAGS)
test_spawn_LDFLAGS =$(AM_LDFLAGS)   $(KAAPIXXPROGS_BUILD_LDFLAGS)
test_spawn_LDADD   =$(LIBS)         $(KAAPIXXPROGS_BUILD_LIBS)

test_pointer_CPPFLAGS=$(AM_CPPFLAGS)  $(KAAPIXXPROGS_BUILD_CPPFLAGS)  
test_pointer_CXXFLAGS=$(AM_CXXFLAGS)  $(KAAPIXXPROGS_BUILD_CXXFLAGS)
test_pointer_LDFLAGS =$(AM_LDFLAGS)   $(KAAPIXXPROGS_BUILD_LDFLAGS)
test_pointer_LDADD   =$(LIBS)         $(KAAPIXXPROGS_BUILD_LIBS)
