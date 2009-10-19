##
##  CMakeLists.txt
##  xkaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009 INRIA. All rights reserved.
##

include_directories (${CMAKE_SOURCE_DIR}/src/ ${CMAKE_BINARY_DIR}/src/ ${CMAKE_SOURCE_DIR}/src/thread ${CMAKE_SOURCE_DIR}/src//workstealing ${CMAKE_SOURCE_DIR}/src/dfg) 
link_directories (${CMAKE_BINARY_DIR}/src/ /usr/lib ) 

add_definitions ( "-Wall -pedantic -g" )
set (CMAKE_C_FLAGS "-std=c99")
set (CKAAPI_TEST_LINK_LIBS xkaapi-static)

# Executable Section :
add_executable (print_sizestruct     print_sizestruct.c) 
add_executable (print_sizestruct_pub print_sizestruct_pub.c)
add_executable (prod-cons            prod-cons.c)
add_executable (test_create1         test_create1.c) 
add_executable (test_create2         test_create2.c)
add_executable (test_mutex1          test_mutex1.c)
add_executable (test_mutex2          test_mutex2.c)
add_executable (test_mutex3          test_mutex3.c)
add_executable (test_mutex4          test_mutex4.c)
add_executable (test_mutex_trylock   test_mutex_trylock.c)
add_executable (test_cond2           test_cond2.c)
add_executable (test_cond3           test_cond3.c)
add_executable (test_cond4           test_cond4.c)
add_executable (test_once            test_once.c)
add_executable (test_key             test_key.c)
add_executable (test_affinity       test_affinity.c)
#add_executable (test_steal1          test_steal1.c )
#add_executable (test_steal2          test_steal2.c )
#add_executable (test_steal3          test_steal3.c )
#add_executable (transform     	      transform.h transform.cpp random.cpp )
#add_executable (find_if              find_if.cpp )
#add_executable (prefix               prefix.cpp random.cpp )
#add_executable (stl_transform        stl_transform.cpp random.cpp )
#add_executable (stl_prefix           stl_prefix.cpp random.cpp )

# Target Section :
target_link_libraries (print_sizestruct     ${CKAAPI_TEST_LINK_LIBS})
target_link_libraries (print_sizestruct_pub ${CKAAPI_TEST_LINK_LIBS})
target_link_libraries (prod-cons            ${CKAAPI_TEST_LINK_LIBS}) 
target_link_libraries (test_create1         ${CKAAPI_TEST_LINK_LIBS}) 
target_link_libraries (test_create2         ${CKAAPI_TEST_LINK_LIBS}) 
target_link_libraries (test_mutex1          ${CKAAPI_TEST_LINK_LIBS}) 
target_link_libraries (test_mutex2          ${CKAAPI_TEST_LINK_LIBS})
target_link_libraries (test_mutex3          ${CKAAPI_TEST_LINK_LIBS})
target_link_libraries (test_mutex4          ${CKAAPI_TEST_LINK_LIBS})
target_link_libraries (test_mutex_trylock   ${CKAAPI_TEST_LINK_LIBS})
target_link_libraries (test_cond2           ${CKAAPI_TEST_LINK_LIBS})
target_link_libraries (test_cond3           ${CKAAPI_TEST_LINK_LIBS})
target_link_libraries (test_cond4           ${CKAAPI_TEST_LINK_LIBS})
target_link_libraries (test_once            ${CKAAPI_TEST_LINK_LIBS})
target_link_libraries (test_key             ${CKAAPI_TEST_LINK_LIBS})
target_link_libraries (test_affinity        ${CKAAPI_TEST_LINK_LIBS} m)
#target_link_libraries (test_steal1          ${CKAAPI_TEST_LINK_LIBS} m)
#target_link_libraries (test_steal2          ${CKAAPI_TEST_LINK_LIBS} m)
#target_link_libraries (test_steal3          ${CKAAPI_TEST_LINK_LIBS} m)
#target_link_libraries (transform            ${CKAAPI_TEST_LINK_LIBS} m)
#target_link_libraries (find_if              ${CKAAPI_TEST_LINK_LIBS} m)
#target_link_libraries (prefix               ${CKAAPI_TEST_LINK_LIBS} m)
#target_link_libraries (stl_transform        m)
#target_link_libraries (stl_prefix           m)
