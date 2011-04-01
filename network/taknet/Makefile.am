##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir)

if KAAPI_TARGET_TAKNET

lib_LTLIBRARIES=libkadev_taknet.la

#BLUEGENE: libkadev_taknet_la_CPPFLAGS=$(AM_CPPFLAGS) -I/usr/local/openmpi/include -I$(top_srcdir)/network -I$(top_srcdir)/api/kaapixx $(KATAKNETLIB_BUILD_CPPFLAGS)
libkadev_taknet_la_CPPFLAGS=$(AM_CPPFLAGS) -I$(top_srcdir)/network -I$(top_srcdir)/api/kaapixx $(KATAKNETLIB_BUILD_CPPFLAGS)
libkadev_taknet_la_CFLAGS=$(AM_CFLAGS) $(KATAKNETLIB_BUILD_CFLAGS)
libkadev_taknet_la_CXXFLAGS=$(AM_CXXFLAGS) $(KATAKNETLIB_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
libkadev_taknet_la_LDFLAGS = -version-info 1:0:1 $(KATAKNETLIB_BUILD_LDFLAGS) $(KATAKNETLIB_BUILD_LIBS)

#BLUEGENE: libkadev_taknet_la_LDFLAGS = -version-info 1:0:1 $(KATAKNETLIB_BUILD_LDFLAGS) -L/usr/local/openmpi/lib -lmpi

#libkadev_taknet_la_LIBADD=../../src/libxkaapi.la $(KATAKNETLIB_BUILD_LIBS)

include_HEADERS=

libkadev_taknet_la_SOURCES=\
	kataknet_init.h \
	kataknet_init.cpp \
	kataknet_device.h \
	kataknet_device.cpp \
	kataknet_devicefactory.cpp \
	kataknet_channel.h \
	kataknet_channel.cpp \
	kataknet_barrier.cpp \
	kataknet_skel.cpp \
	kataknet_alloc.cpp \
	$(include_HEADERS)
endif

check_headers_SRC_C=
check_headers_SRC_CXX=
check_headers_LIBS=$(KATAKNETPROGS_BUILD_LDFLAGS) $(KATAKNETPROGS_BUILD_LIBS) 

