##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir) -I$(srcdir)/../kaapic

if KAAPI_API_KAAPIF

lib_LTLIBRARIES=libkaapif.la

libkaapif_la_CPPFLAGS=$(AM_CPPFLAGS)       $(KAAPIFLIB_BUILD_CPPFLAGS)
libkaapif_la_CFLAGS=  $(AM_CFLAGS)         $(KAAPIFLIB_BUILD_CFLAGS)
libkaapif_la_CXXFLAGS=$(AM_CXXFLAGS)       $(KAAPIFLIB_BUILD_CXXFLAGS)
libkaapif_la_LDFLAGS = -version-info 1:0:1 $(KAAPIFLIB_BUILD_LDFLAGS)
libkaapif_la_LIBADD=../kaapic/libkaapic.la $(KAAPIFLIB_BUILD_LIBS)


M4_BUILT_FILES=\
  kaapif_inc.h \
  kaapif.inc \
  kaapif.inc.f90 \
  kaapif.inc.f95 \
  kaapif_adaptive_switch.h

M4_SRCS=\
  kaapif_adaptive_switch.m4 \
  kaapif_inc.m4 \
  m4_generator \
  m4_macros

BUILT_SOURCES = $(M4_BUILT_FILES)
EXTRA_DIST = $(M4_SRCS)

include_HEADERS = kaapif.inc kaapif.inc.f90 kaapif.inc.f95

$(M4_BUILT_FILES): stamp-m4_generator

.INTERMEDIATE: stamp-m4_generator
stamp-m4_generator: $(M4_SRCS)
	cd $(srcdir) && ./m4_generator 64
	touch $@

clean-local:
	$(RM) stamp-m4_generator

libkaapif_la_SOURCES=\
	kaapif.c \
	kaapif.h\
	kaapif_foreach.c \
	kaapif_spawn.c \
	kaapif_inc.h \
	kaapif_adaptive_switch.h

endif
