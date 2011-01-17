##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir)

if KAAPI_API_KAAPIXX

lib_LTLIBRARIES=libkaapi++.la

libkaapi___la_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIXXLIB_BUILD_CPPFLAGS)
libkaapi___la_CFLAGS=$(AM_CFLAGS) $(KAAPIXXLIB_BUILD_CFLAGS)
libkaapi___la_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIXXLIB_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
libkaapi___la_LDFLAGS = -version-info 1:0:1 $(KAAPIXXLIB_BUILD_LDFLAGS)
libkaapi___la_LIBADD=../../src/libkaapi.la $(KAAPIXXLIB_BUILD_LIBS)

M4_BUILT_FILES=\
	ka_api_tuple.h\
	ka_api_clo.h \
	ka_api_spawn.h\
	ka_api_reqspawn.h\
	ka_api_execgraph.h\
	ka_api_execforeach.h

M4_SRCS=\
	m4_generator \
	m4_macros \
	ka_api_clo.m4 \
	ka_api_spawn.m4\
	ka_api_reqspawn.m4\
	ka_api_execgraph.m4\
	ka_api_execforeach.m4\
	ka_api_tuple_header.m4\
	ka_api_tuple.m4\
	ka_api_tuple_tail.m4

BUILT_SOURCES = $(M4_BUILT_FILES)

EXTRA_DIST=$(M4_BUILT_FILES) $(M4_SRCS)

$(M4_BUILT_FILES): stamp-m4_generator

.INTERMEDIATE: stamp-m4_generator
stamp-m4_generator: $(M4_SRCS)
	cd $(srcdir) && ./m4_generator 10
	touch $@

clean-local:
	$(RM) stamp-m4_generator

include_HEADERS=\
	kaapi++ \
	kaapi++.h \
	ka_error.h \
	ka_timer.h \
	ka_api_array.h \
	ka_api_tuple.h\
	ka_api_clo.h \
	ka_api_spawn.h \
	ka_api_reqspawn.h \
	ka_api_execgraph.h\
	ka_api_execforeach.h

libkaapi___la_SOURCES=\
	ka_init.cpp \
	ka_error.cpp \
	ka_timer.cpp \
	ka_properties.cpp \
	ka_logfile.cpp \
	ka_component.cpp \
	ka_parser.cpp \
	ka_api_format.cpp \
	ka_api_kaapi.cpp \
	ka_api_array.cpp \
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

endif
