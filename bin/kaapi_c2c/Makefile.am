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

kaapi_c2c_SOURCES=\
main.cpp\
globals.cpp\
parser.cpp\
utils.cpp\
kaapi_abort.cpp\
kaapi_loop.cpp\
kaapi_pragma.cpp\
kaapi_c2c_task.cpp\
kaapi_ssa.cpp\
kaapi_taskcall.cpp\
kaapi_initializer.cpp\
kaapi_mode_analysis.cpp\
kaapi_finalization.cpp

kaapi_c2c_CPPFLAGS=$(AM_CPPFLAGS) $(KACC_BUILD_CPPFLAGS) -UNDEBUG
kaapi_c2c_CXXFLAGS=$(AM_CXXFLAGS) $(KACC_BUILD_CXXFLAGS) -UNDEBUG
kaapi_c2c_LDFLAGS =$(AM_LDFLAGS)  $(KACC_BUILD_LDFLAGS) 
kaapi_c2c_LDADD   =$(LIBS)        $(KACC_BUILD_LIBS)
