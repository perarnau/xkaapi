##
##  xkaapi
##
##  Created by VD on 09/11/10.
##  Copyright 2010 INRIA. All rights reserved.
##

BIN_PRGS=kaapi_c2c
bin_PROGRAMS=$(BIN_PRGS)
check_PROGRAMS=$(BIN_PRGS)

all-bfs: $(BIN_PRGS)

kaapi_c2c_SOURCES=kaapi_c2c.cpp 

kaapi_c2c_CPPFLAGS=$(AM_CPPFLAGS) $(KACC_BUILD_CPPFLAGS) -UNDEBUG
kaapi_c2c_CXXFLAGS=$(AM_CXXFLAGS) $(KACC_BUILD_CXXFLAGS) -UNDEBUG
kaapi_c2c_LDFLAGS =$(AM_LDFLAGS)  $(KACC_BUILD_LDFLAGS) 
kaapi_c2c_LDADD   =$(LIBS)        $(KACC_BUILD_LIBS)
