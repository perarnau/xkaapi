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

add_definitions ( "-Wall -pedantic -O3" )
#add_definitions ( "-Wall -pedantic -g -DKAAPI_DEBUG" )
set (CKAAPI_TEST_LINK_LIBS xkaapi-static)

# Executable Section :
add_executable (transform     	      transform.h transform.cpp random.cpp )
#add_executable (find_if              find_if.cpp )
#add_executable (prefix               prefix.cpp random.cpp )
#add_executable (stl_transform        stl_transform.cpp random.cpp )
#add_executable (stl_prefix           stl_prefix.cpp random.cpp )
#add_executable (test_steal1          test_steal1.c )
#add_executable (test_steal2          test_steal2.c )
#add_executable (test_steal3          test_steal3.c )

# Target Section :
target_link_libraries (transform            ${CKAAPI_TEST_LINK_LIBS} m)
#target_link_libraries (find_if              ${CKAAPI_TEST_LINK_LIBS} m)
#target_link_libraries (prefix               ${CKAAPI_TEST_LINK_LIBS} m)
#target_link_libraries (stl_transform        m)
#target_link_libraries (stl_prefix           m)
#target_link_libraries (test_steal1          ${CKAAPI_TEST_LINK_LIBS} m)
#target_link_libraries (test_steal2          ${CKAAPI_TEST_LINK_LIBS} m)
#target_link_libraries (test_steal3          ${CKAAPI_TEST_LINK_LIBS} m)
