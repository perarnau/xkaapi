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

# GLOBAL, LANGUAGE INDEPEND OPTIONS
add_definitions ( "-Wall -g -DKAAPI_DEBUG" )

set (CKAAPI_TEST_LINK_LIBS xkaapi-static)


# NEW TARGET Executable Section :
add_executable (fiboseq               fiboseq.c )
add_executable (fibo                  fibo.c )
add_executable (fibo_adapt            fibo_adapt.c )
add_executable (transform     	      transform.h transform.cpp random.cpp )
add_executable (marc_problem          marc_problem.cpp )


# PROPERTIES for C code
set_target_properties( 
              fiboseq 
              PROPERTIES COMPILE_FLAGS "-std=c99 -DFIBOCODE -DMAINCODE" 
)

# PROPERTIES for C code
set_target_properties( 
              fibo fibo_adapt 
              PROPERTIES COMPILE_FLAGS "-std=c99 -fnested-functions" 
)

# PROPERTIES for C++ code
set_target_properties( 
              transform marc_problem 
              PROPERTIES COMPILE_FLAGS "" 
)


# Target Section :
target_link_libraries (transform            ${CKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (marc_problem         ${CKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (fiboseq              ${CKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (fibo                 ${CKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (fibo_adapt           ${CKAAPI_TEST_LINK_LIBS} m)
