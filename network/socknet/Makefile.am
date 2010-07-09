##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##


AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir)

if KAAPI_TARGET_SOCKNET

lib_LTLIBRARIES=libkasocknet.la

libkasocknet_la_CPPFLAGS=$(AM_CPPFLAGS) $(KASOCKNETLIB_BUILD_CPPFLAGS)
libkasocknet_la_CFLAGS=$(AM_CFLAGS) $(KASOCKNETLIB_BUILD_CFLAGS)
libkasocknet_la_CXXFLAGS=$(AM_CXXFLAGS) $(KASOCKNETLIB_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
libkasocknet_la_LDFLAGS = -version-info 1:0:1 $(KASOCKNETLIB_BUILD_LDFLAGS)
##libkasocknet_la_LIBADD=../libkanet.la $(KASOCKNETLIB_BUILD_LIBS)

include_HEADERS=

libkasocknet_la_SOURCES=\
	kasocknet_device.cpp \
	kasocknet_devicefactory.cpp \
	kasocknet_channel.cpp \
	$(include_HEADERS)

check_headers_SRC_C=
check_headers_SRC_CXX=
check_headers_LIBS=$(KASOCKNETPROGS_BUILD_LDFLAGS) $(KASOCKNETPROGS_BUILD_LIBS)

endif
