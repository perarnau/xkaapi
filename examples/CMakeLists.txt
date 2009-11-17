##
##  CMakeLists.txt
##  xkaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009 INRIA. All rights reserved.
##

# For installed software: include_directories (${CMAKE_INSTALL_PREFIX}/include) 
# For installed software:  link_directories (${CMAKE_INSTALL_PREFIX}/lib/ /usr/lib ) 

include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_BINARY_DIR}../src) 
link_directories (${CMAKE_CURRENT_BINARY_DIR}/../src/ /usr/lib ) 

add_definitions ( "-Wall -O3 -fnested-functions" )
#add_definitions ( "-Wall -pedantic -g -DKAAPI_DEBUG" )
set (CKAAPI_TEST_LINK_LIBS xkaapi-static)

# Executable Section :
add_executable (transform     	      transform.h transform.cpp random.cpp )
add_executable (fibo_adapt            fibo_adapt.c )
add_executable (marc_problem          marc_problem.cpp )

# Target Section :
target_link_libraries (transform            ${CKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (fibo_adapt           ${CKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (marc_problem         ${CKAAPI_TEST_LINK_LIBS} m)
