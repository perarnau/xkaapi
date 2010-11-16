##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir)

if KAAPI_TARGET_MPINET

lib_LTLIBRARIES=libkampinet.la

libkampinet_la_CPPFLAGS=$(AM_CPPFLAGS) -I$(top_srcdir)/api/kaapixx $(KAMPINETLIB_BUILD_CPPFLAGS)
libkampinet_la_CFLAGS=$(AM_CFLAGS) $(KAMPINETLIB_BUILD_CFLAGS)
libkampinet_la_CXXFLAGS=$(AM_CXXFLAGS) $(KAMPINETLIB_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
libkampinet_la_LDFLAGS = -version-info 1:0:1 $(KAMPINETLIB_BUILD_LDFLAGS)
#libkampinet_la_LIBADD=../../src/libxkaapi.la $(KAMPINETLIB_BUILD_LIBS)

include_HEADERS=

libkampinet_la_SOURCES=\
	kampinet_device.cpp \
	kampinet_devicefactory.cpp \
	kampinet_channel.cpp \
	kampinet_skel.cpp \
	$(include_HEADERS)
endif

check_headers_SRC_C=
check_headers_SRC_CXX=
check_headers_LIBS=$(KAMPINETPROGS_BUILD_LDFLAGS) $(KAMPINETPROGS_BUILD_LIBS)

