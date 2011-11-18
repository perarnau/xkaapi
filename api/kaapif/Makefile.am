##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir) -I$(srcdir)/../kaapic

if KAAPI_API_KAAPIF

lib_LTLIBRARIES=libkaapif.la

libkaapif_la_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPILIB_BUILD_CPPFLAGS)
libkaapif_la_CFLAGS=$(AM_CFLAGS) $(KAAPILIB_BUILD_CFLAGS)
libkaapif_la_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPILIB_BUILD_CXXFLAGS)
libkaapif_la_LDFLAGS = -version-info 1:0:1 $(KAAPILIB_BUILD_LDFLAGS)
libkaapif_la_LIBADD=../../src/libkaapi.la $(KAAPILIB_BUILD_LIBS)


M4_BUILT_FILES=\
  kaapif_dfg_switch.h \
  kaapif_adaptive_switch.h

M4_SRCS=\
  kaapif_dfg_switch.m4 \
  kaapif_adaptive_switch.m4 \
  m4_generator \
  m4_macros

BUILT_SOURCES = $(M4_BUILT_FILES)

$(M4_BUILT_FILES): stamp-m4_generator

.INTERMEDIATE: stamp-m4_generator
stamp-m4_generator: $(M4_SRCS)
	cd $(srcdir) && ./m4_generator 32
	touch $@

clean-local:
	$(RM) stamp-m4_generator

include_HEADERS=\
	kaapif.h
    
libkaapif_la_SOURCES=\
	kaapif.c \
	kaapif_foreach.c \
	kaapif_spawn.c \
	$(include_HEADERS)\
    $(M4_BUILT_FILES)

endif