##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir)

if KAAPI_TARGET_GASNET

lib_LTLIBRARIES=libkadev_gasnet.la

#BLUEGENE: libkadev_gasnet_la_CPPFLAGS=$(AM_CPPFLAGS) -I/usr/local/openmpi/include -I$(top_srcdir)/network -I$(top_srcdir)/api/kaapixx $(KAGASNETLIB_BUILD_CPPFLAGS)
libkadev_gasnet_la_CPPFLAGS=$(AM_CPPFLAGS) -I$(top_srcdir)/network -I$(top_srcdir)/api/kaapixx $(KAGASNETLIB_BUILD_CPPFLAGS)
libkadev_gasnet_la_CFLAGS=$(AM_CFLAGS) $(KAGASNETLIB_BUILD_CFLAGS)
libkadev_gasnet_la_CXXFLAGS=$(AM_CXXFLAGS) $(KAGASNETLIB_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
libkadev_gasnet_la_LDFLAGS = -version-info 1:0:1 $(KAGASNETLIB_BUILD_LDFLAGS) $(KAGASNETLIB_BUILD_LIBS)

#BLUEGENE: libkadev_gasnet_la_LDFLAGS = -version-info 1:0:1 $(KAGASNETLIB_BUILD_LDFLAGS) -L/usr/local/openmpi/lib -lmpi

#libkadev_gasnet_la_LIBADD=../../src/libxkaapi.la $(KAGASNETLIB_BUILD_LIBS)

include_HEADERS=

libkadev_gasnet_la_SOURCES=\
	kagasnet_init.h \
	kagasnet_init.cpp \
	kagasnet_device.h \
	kagasnet_device.cpp \
	kagasnet_devicefactory.cpp \
	kagasnet_channel.h \
	kagasnet_channel.cpp \
	kagasnet_skel.cpp \
	kagasnet_alloc.cpp \
	$(include_HEADERS)
endif

check_headers_SRC_C=
check_headers_SRC_CXX=
check_headers_LIBS=$(KAGASNETPROGS_BUILD_LDFLAGS) $(KAGASNETPROGS_BUILD_LIBS) 

