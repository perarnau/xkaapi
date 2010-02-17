##
##  CMakeLists.txt
##  xkaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009 INRIA. All rights reserved.
##

# Add the binary directory for generated headers :
#include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_BINARY_DIR}/../src) 

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src

lib_LTLIBRARIES=libxkaapixx.la

libxkaapixx_la_CPPFLAGS=$(AM_CPPFLAGS) $(XKAAPILIB_BUILD_CPPFLAGS)
libxkaapixx_la_CFLAGS=$(AM_CFLAGS) $(XKAAPILIB_BUILD_CFLAGS)
libxkaapixx_la_CXXFLAGS=$(AM_CXXFLAGS) $(XKAAPILIB_BUILD_CXXFLAGS) $(STD_CXX0X_FLAGS)
libxkaapixx_la_LDFLAGS = -version-info 1:0:1 $(XKAAPILIB_BUILD_LDFLAGS)
libxkaapixx_la_LIBADD=../src/libxkaapi.la $(XKAAPILIB_BUILD_LIBS)

M4_BUILT_FILES=\
	ka_api_clo.h \
	ka_api_fork.h

#	athapascan_closure.h \
#	athapascan_fork.h

M4_SRCS=\
	m4_generator \
	m4_macros \
	ka_api_clo.m4 \
	ka_api_fork.m4

#	athapascan_closure_gen.m4 \
#	athapascan_fork_gen.m4 

BUILT_SOURCES = $(M4_BUILT_FILES)

EXTRA_DIST=$(M4_BUILT_FILES) $(M4_SRCS)

$(M4_BUILT_FILES): $(M4_SRCS)
	$(srcdir)/m4_generator 10

libxkaapixx_la_SOURCES=\
	ka_init.cpp \
	ka_error.cpp \
	ka_timer.cpp \
	ka_properties.cpp \
	ka_logfile.cpp \
	ka_component.cpp \
	ka_parser.cpp \
	ka_api_format.cpp \
	ka_api_kaapi.cpp \
	ka_component.h \
	ka_debug.h \
	ka_error.h \
	ka_init.h \
	ka_parser.h \
	ka_properties.h \
	ka_timer.h \
	ka_types.h\
	atha_kaapi.cpp

# NOT YET USED:  	atha_monotonic.cpp \
#	atha_array.h \
#	ka_format.h \
#	ka_stream.h 

# NOT YET USED:  atha_stream.cpp
#  atha_format.cpp

include_HEADERS=\
	kaapi++ \
	kaapi++.h\
	ka_error.h\
  ka_timer.h\
  ka_api_clo.h\
  ka_api_fork.h
  
# NOT YET USED:  	athapascan-1 \
#	athapascan-1.h 
#  athapascan_closure.h\
#  athapascan_fork.h

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
