##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir) -I$(top_srcdir)/api/kaapic

#if KAAPI_API_LIBGOMP

lib_LTLIBRARIES=liblibgomp.la

liblibgomp_la_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPILIB_BUILD_CPPFLAGS)
liblibgomp_la_CFLAGS=$(AM_CFLAGS) $(KAAPILIB_BUILD_CFLAGS)
liblibgomp_la_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPILIB_BUILD_CXXFLAGS)
liblibgomp_la_LDFLAGS = -version-info 1:0:1 $(KAAPILIB_BUILD_LDFLAGS)
liblibgomp_la_LIBADD=../../src/libkaapi.la $(KAAPILIB_BUILD_LIBS)

include_HEADERS=\
    libgomp.h

liblibgomp_la_SOURCES=\
    libgomp_parallel.c \
    libgomp_barrier.c \
    libgomp_critical.c \
    libgomp_ordered.c \
    libgomp_single.c \
    libgomp_loop.c \
    libgomp_loop_static.c \
    libgomp_section.c \
    libgomp_task.c \
    $(include_HEADERS)

check_headers_SRC_C=
check_headers_SRC_CXX=
check_headers_LIBS=$(KAAPIPROGS_BUILD_LDFLAGS) $(KAAPIPROGS_BUILD_LIBS)

include ../check-headers.am

#endif
