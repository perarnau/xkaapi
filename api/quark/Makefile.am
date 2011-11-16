##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir)

if KAAPI_API_QUARK

lib_LTLIBRARIES=libquark.la

libquark_la_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPILIB_BUILD_CPPFLAGS)
libquark_la_CFLAGS=$(AM_CFLAGS) $(KAAPILIB_BUILD_CFLAGS)
libquark_la_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPILIB_BUILD_CXXFLAGS)
libquark_la_LDFLAGS = -version-info 1:0:1 $(KAAPILIB_BUILD_LDFLAGS)
libquark_la_LIBADD=../../src/libkaapi.la $(KAAPILIB_BUILD_LIBS)

include_HEADERS=\
    quark.h\
    quark_unpack_args.h

libquark_la_SOURCES=\
    quark_xkaapi.c \
	$(include_HEADERS)

check_headers_SRC_C=
check_headers_SRC_CXX=
check_headers_LIBS=$(KAAPIPROGS_BUILD_LDFLAGS) $(KAAPIPROGS_BUILD_LIBS)

include ../check-headers.am

endif
