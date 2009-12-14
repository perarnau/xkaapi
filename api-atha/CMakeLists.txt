##
##  CMakeLists.txt
##  xkaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009 INRIA. All rights reserved.
##

# Add the binary directory for generated headers :
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_BINARY_DIR}/../src) 
#link_directories (${CMAKE_CURRENT_BINARY_DIR}/../src/ /usr/lib ) 
#include (${CMAKE_CURRENT_SOURCE_DIR}/../CheckCCompilerFlag.cmake )

# Global definitions for compile comand :
#add_definitions ( "-Wall -O3")
#add_definitions ( "-Wall -O3 -DKAAPI_CONCURRENT_WS=1")
#add_definitions ( "-Wall -O3 -DKAAPI_USE_PERFCOUNTER=1")
add_definitions ( "-Wall -g -DKAAPI_DEBUG")
#add_definitions ( "-Wall -g")

#################
## LIBRARY BEG ##

# Shared lib :
add_library (xkaapi-cpp SHARED 
  atha_init.cpp
  atha_error.cpp
  atha_timer.cpp
  atha_properties.cpp
  atha_logfile.cpp
  atha_component.cpp
  atha_parser.cpp
  atha_monotonic.cpp
  atha_kaapi.cpp
)
# NOT YET USED:  atha_stream.cpp
#  atha_format.cpp

# Version number :
set_target_properties(xkaapi-cpp PROPERTIES VERSION 1.0.1 SOVERSION 1 OUTPUT_NAME "xkaapi++")
target_link_libraries (xkaapi-cpp           xkaapi)

# Static lib :
add_library (xkaapi-cpp-static STATIC 
  atha_init.cpp
  atha_error.cpp
  atha_timer.cpp
  atha_properties.cpp
  atha_logfile.cpp
  atha_component.cpp
  atha_parser.cpp
  atha_monotonic.cpp
  atha_kaapi.cpp
)
# NOT YET USED:  atha_stream.cpp
#  atha_format.cpp

set_target_properties(xkaapi-cpp-static PROPERTIES OUTPUT_NAME "xkaapi++")
target_link_libraries (xkaapi-cpp-static           xkaapi-static)

# These next two lines are required but it is unclear exactly what they do.
# The CMake FAQ mentions they are necessary and it does not work otherwise.
set_target_properties(xkaapi-cpp PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(xkaapi-cpp-static PROPERTIES CLEAN_DIRECT_OUTPUT 1)

## LIBRARY END ##
#################


#######################
## INSTALL RULES BEG ##

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/athapascan-1 
              ${CMAKE_CURRENT_SOURCE_DIR}/athapascan-1.h
        DESTINATION include/)

## INSTALL RULES END ##
#######################
