##
##  xkaapi
##
##  Created by VD on 09/11/10.
##  Copyright 2010 INRIA. All rights reserved.
##

BIN_PRGS=kacc
bin_PROGRAMS=$(BIN_PRGS)
check_PROGRAMS=$(BIN_PRGS)

all-bfs: $(BIN_PRGS)

kacc_SOURCES=kacc.cpp 
kacc_CPPFLAGS=$(AM_CPPFLAGS) $(KACC_BUILD_CPPFLAGS)
kacc_CXXFLAGS=$(AM_CXXFLAGS)
kacc_LDFLAGS =$(AM_LDFLAGS) 
kacc_LDADD   =$(LIBS)
