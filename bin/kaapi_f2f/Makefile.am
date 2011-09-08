##
##  xkaapi
##
##  Created by VD on 09/11/10.
##  Copyright 2010 INRIA. All rights reserved.
##

BIN_PRGS=kaapi_f2f
bin_PROGRAMS=$(BIN_PRGS)
check_PROGRAMS=$(BIN_PRGS)

all-bfs: $(BIN_PRGS)

kaapi_f2f_SOURCES=\
main.cpp

kaapi_f2f_CPPFLAGS=$(AM_CPPFLAGS) $(KACC_BUILD_CPPFLAGS) -UNDEBUG
kaapi_f2f_CXXFLAGS=$(AM_CXXFLAGS) $(KACC_BUILD_CXXFLAGS) -UNDEBUG
kaapi_f2f_LDFLAGS =$(AM_LDFLAGS)  $(KACC_BUILD_LDFLAGS) 
kaapi_f2f_LDADD   =$(LIBS)        $(KACC_BUILD_LIBS)
