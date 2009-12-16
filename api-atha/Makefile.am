##
##  CMakeLists.txt
##  xkaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009 INRIA. All rights reserved.
##

# Add the binary directory for generated headers :
#include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_BINARY_DIR}/../src) 

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(srcdir)/../src -I$(srcdir)/../src/machine/$(KAAPI_MACHINE)

lib_LTLIBRARIES=libatha1.la

libatha1_la_CPPFLAGS=$(AM_CPPFLAGS) $(ATHA1_BUILD_CPPFLAGS) -pedantic
libatha1_la_CFLAGS=$(AM_CFLAGS) $(ATHA1_BUILD_CFLAGS)
libatha1_la_CXXFLAGS=$(AM_CXXFLAGS) $(ATHA1_BUILD_CXXFLAGS) -std=gnu++0x
libatha1_la_LDFLAGS = -version-info 1:0:1 $(ATHA1_BUILD_LDFLAGS)
libatha1_la_LIBADD=../src/libxkaapi.la $(ATHA1_BUILD_LIBS)

libatha1_la_SOURCES=\
	atha_init.cpp \
	atha_error.cpp \
	atha_timer.cpp \
	atha_properties.cpp \
	atha_logfile.cpp \
	atha_component.cpp \
	atha_parser.cpp \
	atha_monotonic.cpp \
	atha_kaapi.cpp \
	atha_array.h \
	atha_component.h \
	atha_debug.h \
	atha_error.h \
	atha_format.h \
	atha_init.h \
	atha_parser.h \
	athapascan-1.h \
	athapascan_closure.h \
	athapascan_fork.h \
	atha_properties.h \
	atha_stream.h \
	atha_timer.h \
	atha_types.h

# NOT YET USED:  atha_stream.cpp
#  atha_format.cpp

include_HEADERS=\
	athapascan-1 \
	athapascan-1.h

check_PROGRAMS=atha1_check_header atha1_check_header_strict
check_headers_SRC=\
        check/atha1_header.cpp \
        check/atha1_h_header.cpp \
        check/empty_main.c

atha1_check_header_strict_CPPFLAGS=$(AM_CPPFLAGS) -Werror -pedantic
atha1_check_header_strict_CXXFLAGS=$(AM_CXXFLAGS) -std=c++0x
atha1_check_header_strict_SOURCES=\
        $(check_headers_SRC)

# No -pedantic here...
atha1_check_header_CPPFLAGS=$(AM_CPPFLAGS) -Werror -D_KAAPI_DISABLE_WARNINGS
atha1_check_header_SOURCES=\
        $(check_headers_SRC)
