##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##
AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir)

SUBDIRS= . socknet mpinet

if KAAPI_TARGET_KANET

lib_LTLIBRARIES=libkanet.la

libkanet_la_CPPFLAGS=$(AM_CPPFLAGS) $(KANETLIB_BUILD_CPPFLAGS)
libkanet_la_CFLAGS=$(AM_CFLAGS) $(KANETLIB_BUILD_CFLAGS)
libkanet_la_CXXFLAGS=$(AM_CXXFLAGS) $(KANETLIB_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
libkanet_la_LDFLAGS = -version-info 1:0:1 $(KANETLIB_BUILD_LDFLAGS)
#libkanet_la_LIBADD=../../src/libxkaapi.la $(KANETLIB_BUILD_LIBS)

include_HEADERS=\
	kanet_instr.h \
	kanet_channel.h \
	kanet_device.h

libkanet_la_SOURCES=\
	kanet_instr.cpp \
	kanet_channel.cpp \
	kanet_device.cpp \
	$(include_HEADERS)

check_headers_SRC_C=
check_headers_SRC_CXX=
check_headers_LIBS=$(KANETPROGS_BUILD_LDFLAGS) $(KANETPROGS_BUILD_LIBS)

endif