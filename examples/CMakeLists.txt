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
#add_definitions ( "-Wall -O3")
#add_definitions ( "-Wall -O3 -DKAAPI_CONCURRENT_WS=1")
#add_definitions ( "-Wall -g")
add_definitions ( "-Wall -g -DKAAPI_DEBUG")

set (XKAAPI_TEST_LINK_LIBS xkaapi-static)
set (XKAAPI_TEST_CPP_LINK_LIBS xkaapi-cpp-static)


# NEW TARGET Executable Section :
add_executable (fiboseq               fiboseq.c )
add_executable (fibo                  fibo.c )
add_executable (fibo_adapt            fibo_adapt.c )
add_executable (fibo_apiatha          fibo_apiatha.cpp )
add_executable (nqueens_apiatha       nqueens_apiatha.cpp )
add_executable (branchbound_apiatha   branchbound_apiatha.cpp )
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
  PROPERTIES COMPILE_FLAGS "-std=c99" 
)

if (HAS_NESTED_FUNCTIONS_FLAG)
  set_target_properties(
    fibo fibo_adapt 
    PROPERTIES COMPILE_FLAGS "-std=c99 -fnested-functions"
    )
endif(HAS_NESTED_FUNCTIONS_FLAG)


# PROPERTIES for C++ code
set_target_properties( 
              transform marc_problem fibo_apiatha nqueens_apiatha branchbound_apiatha
              PROPERTIES COMPILE_FLAGS "" 
)


# Target Section :
target_link_libraries (transform            ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (fiboseq              ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (fibo                 ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (fibo_adapt           ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (fibo_apiatha         ${XKAAPI_TEST_CPP_LINK_LIBS} ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (nqueens_apiatha      ${XKAAPI_TEST_CPP_LINK_LIBS} ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (branchbound_apiatha  ${XKAAPI_TEST_CPP_LINK_LIBS} ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (marc_problem         ${XKAAPI_TEST_CPP_LINK_LIBS} ${XKAAPI_TEST_LINK_LIBS} m)

