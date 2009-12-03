##
##  CMakeLists.txt
##  xkaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009 INRIA. All rights reserved.
##

# For installed software: include_directories (${CMAKE_INSTALL_PREFIX}/include) 
# For installed software:  link_directories (${CMAKE_INSTALL_PREFIX}/lib/ /usr/lib ) 

include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_SOURCE_DIR}/../api-atha ${CMAKE_CURRENT_BINARY_DIR}../src) 
link_directories (${CMAKE_CURRENT_BINARY_DIR}/../src/ /usr/lib ) 
include (${CMAKE_CURRENT_SOURCE_DIR}/../CheckCCompilerFlag.cmake )

CHECK_C_COMPILER_FLAG("-fnested-functions" HAS_NESTED_FUNCTIONS_FLAG)

# GLOBAL, LANGUAGE INDEPEND OPTIONS
add_definitions ( "-Wall -g" )
#add_definitions ( "-Wall -g -DKAAPI_DEBUG" )
#add_definitions ( "-Wall -O3" )

set (XKAAPI_TEST_LINK_LIBS xkaapi-static)
set (XKAAPI_TEST_CPP_LINK_LIBS xkaapi-cpp-static)


# NEW TARGET Executable Section :
add_executable (transform     	      transform.h transform.cpp random.cpp kaapi_utils.cpp )
add_executable (accumulate     	      accumulate.h accumulate.cpp random.cpp kaapi_utils.cpp )
add_executable (copy     	      copy.h copy.cpp random.cpp kaapi_utils.cpp )


# PROPERTIES for C++ code
set_target_properties( 
              transform
              accumulate
              copy
              PROPERTIES COMPILE_FLAGS "" 
)


# Target Section :
target_link_libraries (transform            ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (accumulate           ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (copy                 ${XKAAPI_TEST_LINK_LIBS} m)
