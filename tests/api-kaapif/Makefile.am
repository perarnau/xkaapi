##
##  xkaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009,2010,2011,2012 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -Werror            $(KAAPIFPROGS_BUILD_CPPFLAGS)
AM_CFLAGS=  @AM_CFLAGS@   $(STD_C99_FLAGS)   $(KAAPIFPROGS_BUILD_CFLAGS)
AM_CXXFLAGS=@AM_CXXFLAGS@ $(STD_CXX0X_FLAGS) $(KAAPIFPROGS_BUILD_CXXFLAGS)
AM_LDFLAGS= @AM_LDFLAGS@                     $(KAAPIFPROGS_BUILD_LDFLAGS)
AM_LIBS=    @AM_LIBS@                        $(KAAPIFPROGS_BUILD_LIBS)

noinst_PROGRAMS=

if  KAAPI_API_KAAPIF

noinst_PROGRAMS+=   \
  spawn_v-args
endif

spawn_v_args_SOURCES=spawn_v-args_main.F spawn_v-args_bar.f spawn_v-args_fu.f

