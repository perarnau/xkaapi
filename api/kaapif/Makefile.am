##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir)

if KAAPI_API_KAAPIF

lib_LTLIBRARIES=libkaapif.la

libkaapif_la_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPILIB_BUILD_CPPFLAGS)
libkaapif_la_CFLAGS=$(AM_CFLAGS) $(KAAPILIB_BUILD_CFLAGS)
libkaapif_la_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPILIB_BUILD_CXXFLAGS)
libkaapif_la_LDFLAGS = -version-info 1:0:1 $(KAAPILIB_BUILD_LDFLAGS)
libkaapif_la_LIBADD=../../src/libkaapi.la $(KAAPILIB_BUILD_LIBS)

include_HEADERS=\
	kaapif.h\
	kaapic.h

libkaapif_la_SOURCES=\
	kaapif.c \
	$(include_HEADERS)

endif