##
##  CMakeLists.txt
##  xkaapi
##
##  Created by CL and TG on 04/02/09.
##  Updated by FLM 02/12/09.
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
add_executable (count     	      count.h count.cpp random.cpp kaapi_utils.cpp )
add_executable (count_if     	      count_if.h count_if.cpp random.cpp kaapi_utils.cpp )
add_executable (fill     	      fill.h fill.cpp random.cpp kaapi_utils.cpp )
add_executable (find_if     	      find_if.h find_if.cpp random.cpp kaapi_utils.cpp )
add_executable (for_each     	      for_each.h for_each.cpp random.cpp kaapi_utils.cpp )
add_executable (generate     	      generate.h generate.cpp random.cpp kaapi_utils.cpp )
add_executable (inner_product 	      inner_product.h inner_product.cpp random.cpp kaapi_utils.cpp )
add_executable (merge		      merge.h merge.cpp random.cpp kaapi_utils.cpp )
add_executable (max_element 	      max_element.h max_element.cpp random.cpp kaapi_utils.cpp )
add_executable (min_element 	      min_element.h min_element.cpp random.cpp kaapi_utils.cpp )
add_executable (replace_copy 	      replace_copy.h replace_copy.cpp random.cpp kaapi_utils.cpp )
add_executable (replace_copy_if	      replace_copy_if.h replace_copy_if.cpp random.cpp kaapi_utils.cpp )
add_executable (replace		      replace.h replace.cpp random.cpp kaapi_utils.cpp )
add_executable (replace_if	      replace_if.h replace_if.cpp random.cpp kaapi_utils.cpp )
add_executable (sort		      sort.h sort.cpp random.cpp kaapi_utils.cpp )
add_executable (stable_sort	      stable_sort.h stable_sort.cpp random.cpp kaapi_utils.cpp )


# PROPERTIES for C++ code
set_target_properties( 
              transform
              accumulate
              copy
              count
              count_if
              fill
              find_if
              for_each
              generate
	      inner_product
	      merge
	      max_element
	      min_element
	      replace_copy
	      replace_copy_if
	      replace
	      replace_if
	      sort
	      stable_sort
              PROPERTIES COMPILE_FLAGS "" 
)


# Target Section :
target_link_libraries (transform            ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (accumulate           ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (copy                 ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (count                ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (count_if             ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (fill                 ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (find_if              ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (for_each             ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (generate             ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (inner_product        ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (merge                ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (max_element          ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (min_element          ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (replace_copy         ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (replace_copy_if      ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (replace		    ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (replace_if	    ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (sort		    ${XKAAPI_TEST_LINK_LIBS} m)
target_link_libraries (stable_sort	    ${XKAAPI_TEST_LINK_LIBS} m)
