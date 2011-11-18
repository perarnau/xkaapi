##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir)

if KAAPI_API_KAAPIC

lib_LTLIBRARIES=libkaapic.la

libkaapic_la_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPILIB_BUILD_CPPFLAGS)
libkaapic_la_CFLAGS=$(AM_CFLAGS) $(KAAPILIB_BUILD_CFLAGS)
libkaapic_la_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPILIB_BUILD_CXXFLAGS)
libkaapic_la_LDFLAGS = -version-info 1:0:1 $(KAAPILIB_BUILD_LDFLAGS)
libkaapic_la_LIBADD=../../src/libkaapi.la $(KAAPILIB_BUILD_LIBS)

M4_BUILT_FILES=\
  kaapic_dfg_switch.h \
  kaapic_adaptive_switch.h

M4_SRCS=\
  kaapic_dfg_switch.m4 \
  kaapic_adaptive_switch.m4 \
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
	kaapic.h

libkaapic_la_SOURCES=\
	kaapic.c \
	kaapic_spawn.c \
	kaapic_work.c \
	kaapic_foreach_docall.c \
	kaapic_foreach.c \
	kaapic_foreach_withformat.c \
	$(include_HEADERS)\
    $(M4_BUILT_FILES)

endif