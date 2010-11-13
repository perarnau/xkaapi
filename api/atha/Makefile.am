##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir) -I$(srcdir)/../kaapixx

lib_LTLIBRARIES=libatha.la

libatha_la_CPPFLAGS=$(AM_CPPFLAGS) $(ATHALIB_BUILD_CPPFLAGS)
libatha_la_CFLAGS=$(AM_CFLAGS) $(ATHALIB_BUILD_CFLAGS)
libatha_la_CXXFLAGS=$(AM_CXXFLAGS) $(ATHALIB_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
libatha_la_LDFLAGS = -version-info 1:0:1 $(ATHALIB_BUILD_LDFLAGS)
libatha_la_LIBADD=../kaapixx/libkaapi++.la $(ATHALIB_BUILD_LIBS)

M4_BUILT_FILES=\
	athapascan_closure.h \
	athapascan_fork.h

M4_SRCS=\
	m4_generator \
	m4_macros \
	athapascan_closure_gen.m4 \
	athapascan_fork_gen.m4

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
	athapascan-1 \
	athapascan-1.h \
	athapascan_closure.h\
	athapascan_fork.h

libatha_la_SOURCES=\
	atha_kaapi.cpp \
	atha_monotonic.cpp \
	$(include_HEADERS)

check_headers_SRC_C=
check_headers_SRC_CXX=\
        check/athapascan-1.cpp \
        check/athapascan-1_h.cpp
check_headers_LIBS=$(ATHAPROGS_BUILD_LDFLAGS) $(ATHAPROGS_BUILD_LIBS)

include ../check-headers.am

