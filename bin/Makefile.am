##
##  xkaapi
##
##  Created by VD on 09/11/10.
##  Copyright 2010 INRIA. All rights reserved.
##

if KAAPI_TARGET_KACC
KAAPI_CC_BIN_PRG=kaapi_c2c kacc
endif

BIN_PRGS=kaapi_event_reader $(KAAPI_CC_BIN_PRG)
bin_PROGRAMS=$(BIN_PRGS)
check_PROGRAMS=$(BIN_PRGS)

all-bfs: $(BIN_PRGS)

kaapi_event_reader_SOURCES=kaapi_event_reader.cpp PajeGenerator.cpp PajeGenerator.hpp

kaapi_event_reader_CPPFLAGS=$(AM_CPPFLAGS) $(KAAPIPROGS_BUILD_CPPFLAGS)
kaapi_event_reader_CXXFLAGS=$(AM_CXXFLAGS) $(KAAPIPROGS_BUILD_CXXFLAGS)
kaapi_event_reader_LDFLAGS =$(AM_LDFLAGS)  $(KAAPIPROGS_BUILD_LDFLAGS)
kaapi_event_reader_LDADD   =$(LIBS)        $(KAAPIPROGS_BUILD_LIBS)

kaapi_c2c_SOURCES=kaapi_c2c.cpp 
kaapi_c2c_CPPFLAGS=$(AM_CPPFLAGS) $(KACC_BUILD_CPPFLAGS) -UNDEBUG
kaapi_c2c_CXXFLAGS=$(AM_CXXFLAGS) $(KACC_BUILD_CXXFLAGS) -UNDEBUG
kaapi_c2c_LDFLAGS =$(AM_LDFLAGS)  $(KACC_BUILD_LDFLAGS) 
kaapi_c2c_LDADD   =$(LIBS)        $(KACC_BUILD_LIBS)

kacc_SOURCES=kacc.cpp 
kacc_CPPFLAGS=$(AM_CPPFLAGS) $(KACC_BUILD_CPPFLAGS)
kacc_CXXFLAGS=$(AM_CXXFLAGS)
kacc_LDFLAGS =$(AM_LDFLAGS) 
kacc_LDADD   =$(LIBS)
