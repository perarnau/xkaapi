##
##  Created by VD on 16/04/2010.
##  Copyright 2010 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(srcdir)

# Not object in the library !!!
#lib_LTLIBRARIES=libkastl.la

libkastl_la_CPPFLAGS=$(AM_CPPFLAGS) $(KASTLLIB_BUILD_CPPFLAGS)
libkastl_la_CFLAGS=$(AM_CFLAGS) $(KASTLLIB_BUILD_CFLAGS)
libkastl_la_CXXFLAGS=$(AM_CXXFLAGS) $(KASTLLIB_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
libkastl_la_LDFLAGS = -version-info 1:0:1 $(KASTLLIB_BUILD_LDFLAGS)
libkastl_la_LIBADD=../../src/libxkaapi.la $(KASTLLIB_BUILD_LIBS)

kastlincludedir=$(includedir)/kastl

kastlinclude_HEADERS=\
	kastl/kastl_types.h\
	kastl/kastl_fifoqueue.h\
	kastl/kastl_workqueue.h\
	kastl/kastl_team.h\
	kastl/kastl_foreach.h\
	kastl/kastl_transform.h\
	kastl/kastl_accumulate_while.h\
	kastl/algorithm \
	kastl/numeric

libkastl_la_SOURCES=\
	$(kastlinclude_HEADERS)

#Put in .h 	kastl_workqueue.cpp 
