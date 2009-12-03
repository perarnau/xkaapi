##
##  CMakeLists.txt
##  xkaapi
##
##  Created by CL, TG and VD on 04/02/09.
##  Copyright 2009 INRIA. All rights reserved.
##

# CMake minimum version :
cmake_minimum_required (VERSION 2.6) 
# Project name :
project (XKAAPI)
set (XKAAPI_VERSION "0.1")
set (XKAAPI_NAME    "X-KAAPI")

###################
## TEST AREA BEG ##

# Allow the "make check" target :
enable_testing ()
set (TESTDIR ${CMAKE_BINARY_DIR}/tests)

#        (testname           exename                     arg)
add_test ("create_process"   ${TESTDIR}/test_create1    1000)
add_test ("create_system"    ${TESTDIR}/test_create2     100)
add_test ("cond2_process"    ${TESTDIR}/test_cond2         1)
add_test ("cond2_system"     ${TESTDIR}/test_cond2         2)
add_test ("cond3_process"    ${TESTDIR}/test_cond3         1)
add_test ("cond3_system"     ${TESTDIR}/test_cond3         2)
add_test ("cond4_process"    ${TESTDIR}/test_cond4         1)
add_test ("cond4_system"     ${TESTDIR}/test_cond4         2)
add_test ("keys_process"     ${TESTDIR}/test_key           1)
add_test ("keys_system"      ${TESTDIR}/test_key           2)
add_test ("once_process"     ${TESTDIR}/test_once          1)
add_test ("once_system"      ${TESTDIR}/test_once          2)
add_test ("mutex1"           ${TESTDIR}/test_mutex1      200)
add_test ("mutex2"           ${TESTDIR}/test_mutex2      200)
add_test ("mutex3"           ${TESTDIR}/test_mutex3      200)
add_test ("mutex4_process"   ${TESTDIR}/test_mutex4        1)
add_test ("mutex4_system"    ${TESTDIR}/test_mutex4        2)
add_test ("trylock_process"  ${TESTDIR}/test_mutex_trylock 1)
add_test ("trylock_system"   ${TESTDIR}/test_mutex_trylock 2)
add_test ("prodcons_process" ${TESTDIR}/prod-cons          1)
add_test ("prodcons_system"  ${TESTDIR}/prod-cons          2)

# Forces dependencies for "make check" :
add_custom_target (check COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS
  test_create1 test_create2 test_cond2 test_cond3 test_cond4 test_key
  test_once test_mutex1 test_mutex2 test_mutex3 test_mutex4
  test_mutex_trylock prod-cons)

## TEST AREA END ##
###################

######################
## PACKAGE AREA BEG ##

include(InstallRequiredSystemLibraries)

macro(cpack_optional_append _list _cond _item)
  if(${_cond})
    set(${_list} ${${_list}} ${_item})
  endif(${_cond})
endmacro(cpack_optional_append _list _cond _item)

set (CPACK_DEBIAN_PACKAGE_MAINTAINER "christophe laferriere")
set (CPACK_DEBIAN_PACKAGE_NAME "libxkaapi-dev")
set (CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${ARCH})

if(NOT CPACK_GENERATOR)
  if(UNIX)
    if(APPLE)
      set(CPACK_BINARY_PACKAGEMAKER  ON)
      set(CPACK_BINARY_OSXX11        OFF)
    else(APPLE)
      set(CPACK_BINARY_TZ            OFF)
    endif(APPLE)
    set(CPACK_BINARY_STGZ            ON)
    set(CPACK_BINARY_TGZ             ON)
    set(CPACK_BINARY_TBZ2            OFF)
    set(CPACK_BINARY_DEB             ON)
    set(CPACK_BINARY_RPM             OFF)
  else(UNIX)
    set(CPACK_BINARY_ZIP             ON)
  endif(UNIX)

  cpack_optional_append(CPACK_GENERATOR  CPACK_BINARY_PACKAGEMAKER PackageMaker)
  cpack_optional_append(CPACK_GENERATOR  CPACK_BINARY_OSXX11       OSXX11)
  cpack_optional_append(CPACK_GENERATOR  CPACK_BINARY_DEB          DEB)
  cpack_optional_append(CPACK_GENERATOR  CPACK_BINARY_RPM          RPM)
  cpack_optional_append(CPACK_GENERATOR  CPACK_BINARY_STGZ         STGZ)
  cpack_optional_append(CPACK_GENERATOR  CPACK_BINARY_TGZ          TGZ)
  cpack_optional_append(CPACK_GENERATOR  CPACK_BINARY_TBZ2         TBZ2)
  cpack_optional_append(CPACK_GENERATOR  CPACK_BINARY_TZ           TZ)
  cpack_optional_append(CPACK_GENERATOR  CPACK_BINARY_ZIP          ZIP)
  
endif(NOT CPACK_GENERATOR)

set (CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.txt")
set (CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.txt")
set (CPACK_PACKAGE_DESCRIPTION_SUMMARY "X-Kaapi is a C library that allows to
  execute multithreaded computation with data flow synchronization between
  tasks")

#set (CPACK_PACKAGE_INSTALL_DIRECTORY "/usr/local")
#set (CPACK_PACKAGE_INSTALL_DIRECTORY "CMake ${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}")

set (CPACK_PACKAGE_NAME "libxkaapi")
set (CPACK_PACKAGE_VENDOR "INRIA")
set (CPACK_PACKAGE_VERSION_MAJOR "1")
set (CPACK_PACKAGE_VERSION_MINOR "0")
set (CPACK_PACKAGE_VERSION_PATCH "0")
set (CPACK_SOURCE_GENERATOR "TGZ")

file (GLOB DOT ".*")
file (GLOB TILD "*~")
set (CPACK_SOURCE_IGNORE_FILES "${DOT};${TILD}")

set (CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENCE.txt")

include (CPack)

## PACKAGE AREA END ##
######################

add_subdirectory (src)
add_subdirectory (api-atha)
#add_subdirectory (examples)
add_subdirectory (kastl.wip)
#TODO TG add_subdirectory (doc)
