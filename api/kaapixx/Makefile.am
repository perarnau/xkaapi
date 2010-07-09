##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir)

lib_LTLIBRARIES=libkaapixx.la

libkaapixx_la_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXLIB_BUILD_CPPFLAGS)
libkaapixx_la_CFLAGS=$(AM_CFLAGS) $(KAAPIXXLIB_BUILD_CFLAGS)
libkaapixx_la_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXLIB_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
libkaapixx_la_LDFLAGS = -version-info 1:0:1 $(KAAPIXXLIB_BUILD_LDFLAGS)
libkaapixx_la_LIBADD=../../src/libxkaapi.la $(KAAPIXXLIB_BUILD_LIBS)

M4_BUILT_FILES=\
	ka_api_clo.h \
	ka_api_spawn.h\
	ka_api_execgraph.h\
	ka_api_execforeach.h

M4_SRCS=\
	m4_generator \
	m4_macros \
	ka_api_clo.m4 \
	ka_api_spawn.m4\
	ka_api_execgraph.m4\
	ka_api_execforeach.m4

BUILT_SOURCES = $(M4_BUILT_FILES)

EXTRA_DIST=$(M4_BUILT_FILES) $(M4_SRCS)

$(M4_BUILT_FILES): $(M4_SRCS)
	(cd $(srcdir); ./m4_generator 10)

include_HEADERS=\
	kaapi++ \
	kaapi++.h \
	ka_error.h \
	ka_timer.h \
	ka_api_clo.h \
	ka_api_execgraph.h\
	ka_api_execforeach.h\
	ka_api_spawn.h

libkaapixx_la_SOURCES=\
	ka_init.cpp \
	ka_error.cpp \
	ka_timer.cpp \
	ka_properties.cpp \
	ka_logfile.cpp \
	ka_component.cpp \
	ka_parser.cpp \
	ka_api_format.cpp \
	ka_api_kaapi.cpp \
	ka_component.h \
	ka_debug.h \
	ka_error.h \
	ka_init.h \
	ka_parser.h \
	ka_properties.h \
	ka_timer.h \
	ka_types.h\
	$(include_HEADERS)

check_headers_SRC_C=
check_headers_SRC_CXX=\
	check/kaapixx.cpp \
	check/kaapixx_h.cpp
check_headers_LIBS=$(KAAPIXXPROGS_BUILD_LDFLAGS) $(KAAPIXXPROGS_BUILD_LIBS)

include ../check-headers.am

