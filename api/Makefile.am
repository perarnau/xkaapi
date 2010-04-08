##
##  CMakeLists.txt
##  xkaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009 INRIA. All rights reserved.
##

# Add the binary directory for generated headers :
#include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_BINARY_DIR}/../src) 

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(top_srcdir)/api/kaapixx 

lib_LTLIBRARIES=libxkaapixx.la

libxkaapixx_la_CPPFLAGS=$(AM_CPPFLAGS) $(XKAAPILIB_BUILD_CPPFLAGS)
libxkaapixx_la_CFLAGS=$(AM_CFLAGS) $(XKAAPILIB_BUILD_CFLAGS)
libxkaapixx_la_CXXFLAGS=$(AM_CXXFLAGS) $(XKAAPILIB_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
libxkaapixx_la_LDFLAGS = -version-info 1:0:1 $(XKAAPILIB_BUILD_LDFLAGS)
libxkaapixx_la_LIBADD=../src/libxkaapi.la $(XKAAPILIB_BUILD_LIBS)

M4_BUILT_FILES=\
	kaapixx/ka_api_clo.h \
	kaapixx/ka_api_fork.h \
	atha/athapascan_closure.h \
	atha/athapascan_fork.h

M4_SRCS=\
	kaapixx/m4_generator \
	kaapixx/m4_macros \
	kaapixx/ka_api_clo.m4 \
	kaapixx/ka_api_fork.m4 \
	atha/m4_generator \
	atha/m4_macros \
	atha/athapascan_closure_gen.m4 \
	atha/athapascan_fork_gen.m4 

BUILT_SOURCES = $(M4_BUILT_FILES)

EXTRA_DIST=$(M4_BUILT_FILES) $(M4_SRCS)

$(M4_BUILT_FILES): $(M4_SRCS)
	(cd $(srcdir)/kaapixx; ./m4_generator 10); 	(cd $(srcdir)/atha; ./m4_generator 10)

libxkaapixx_la_SOURCES=\
	kaapixx/ka_init.cpp \
	kaapixx/ka_error.cpp \
	kaapixx/ka_timer.cpp \
	kaapixx/ka_properties.cpp \
	kaapixx/ka_logfile.cpp \
	kaapixx/ka_component.cpp \
	kaapixx/ka_parser.cpp \
	kaapixx/ka_api_format.cpp \
	kaapixx/ka_api_kaapi.cpp \
	kaapixx/ka_component.h \
	kaapixx/ka_debug.h \
	kaapixx/ka_error.h \
	kaapixx/ka_init.h \
	kaapixx/ka_parser.h \
	kaapixx/ka_properties.h \
	kaapixx/ka_timer.h \
	kaapixx/ka_types.h\
  atha/atha_kaapi.cpp\
  kastl/kastl_workqueue.cpp

# NOT YET USED:  	atha_monotonic.cpp \
#	atha_array.h \
#	ka_format.h \
#	ka_stream.h 

# NOT YET USED:  atha_stream.cpp
#  atha_format.cpp

include_HEADERS=\
	kaapixx/kaapi++ \
	kaapixx/kaapi++.h\
	kaapixx/ka_error.h\
  kaapixx/ka_timer.h\
  kaapixx/ka_api_clo.h\
  kaapixx/ka_api_fork.h \
  atha/athapascan-1 \
  atha/athapascan-1.h \
  atha/athapascan_closure.h\
  atha/athapascan_fork.h
  
kastlinclude_HEADERS=\  
  kastl/kastl_workqueue.h\
  kastl/kastl_foreach.h\
  kastl/kastl_transform.h\
  kastl/algorithm.h\
  kastl/algorithm
kastlincludedir=$(includedir)/kastl

check_PROGRAMS=ka_api_check_header
if STD_CXX0X
check_PROGRAMS+=atha1_check_header_strict
endif
check_headers_SRC=\
        check/ka_api_header.cpp \
        check/ka_api_h_header.cpp \
        check/empty_main.c

# NOT YET USED:       check/atha1_header.cpp \
        check/atha1_h_header.cpp 

ka_api_check_header_strict_CPPFLAGS=$(AM_CPPFLAGS) -Werror
ka_api_check_header_strict_CXXFLAGS=$(AM_CXXFLAGS) $(STD_CXX0X_FLAGS)
ka_api_check_header_strict_SOURCES=\
        $(check_headers_SRC)

# No -pedantic here...
ka_api_check_header_CPPFLAGS=$(AM_CPPFLAGS) -Werror -D_KAAPI_DISABLE_WARNINGS
ka_api_check_header_SOURCES=\
        $(check_headers_SRC)
