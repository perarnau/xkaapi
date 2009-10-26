##
##  CMakeLists.txt
##  ckaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009 INRIA. All rights reserved.
##

FIND_PACKAGE(Doxygen)

IF (DOXYGEN_FOUND)

  # click+jump in Emacs and Visual Studio (for doxy.config) (jw)
  IF    (CMAKE_BUILD_TOOL MATCHES "(msdev|devenv)")
    SET(DOXY_WARN_FORMAT "\"$file($line) : $text \"")
  ELSE  (CMAKE_BUILD_TOOL MATCHES "(msdev|devenv)")
    SET(DOXY_WARN_FORMAT "\"$file:$line: $text \"")
  ENDIF (CMAKE_BUILD_TOOL MATCHES "(msdev|devenv)")
  
  # we need latex for doxygen because of the formulas
  FIND_PACKAGE(LATEX)
  
  IF    (NOT LATEX_COMPILER)
    MESSAGE("++ Check for latex - not found")
  ENDIF (NOT LATEX_COMPILER)
  
  IF    (NOT MAKEINDEX_COMPILER)
    MESSAGE("++ Check for makeindex - not found")
  ENDIF (NOT MAKEINDEX_COMPILER)
  IF    (NOT DVIPS_CONVERTER)
    MESSAGE("++ Check for dvips - not found")
  ENDIF (NOT DVIPS_CONVERTER)
  
  IF   (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/doxy.config.in")
    #MESSAGE(STATUS "configured ${CMAKE_CURRENT_SOURCE_DIR}/doxy.config.in --> ${CMAKE_CURRENT_BINARY_DIR}/doxy.config")
    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/doxy.config.in 
      ${CMAKE_CURRENT_BINARY_DIR}/doxy.config
      @ONLY )
    # use (configured) doxy.config from (out of place) BUILD tree:
    SET(DOXY_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/doxy.config")
  ELSE (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/doxy.config.in")
    # use static hand-edited doxy.config from SOURCE tree:
    SET(DOXY_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/doxy.config")
    IF   (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/doxy.config")
      MESSAGE(STATUS "WARNING: using existing ${CMAKE_CURRENT_SOURCE_DIR}/doxy.config instead of configuring from doxy.config.in file.")
    ELSE (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/doxy.config")
      IF   (EXISTS "${CMAKE_MODULE_PATH}/doxy.config.in")
        # using template doxy.config.in
        #MESSAGE(STATUS "configured ${CMAKE_CMAKE_MODULE_PATH}/doxy.config.in --> ${CMAKE_CURRENT_BINARY_DIR}/doxy.config")
        CONFIGURE_FILE(${CMAKE_MODULE_PATH}/doxy.config.in 
          ${CMAKE_CURRENT_BINARY_DIR}/doxy.config
          @ONLY )
        SET(DOXY_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/doxy.config")
      ELSE (EXISTS "${CMAKE_MODULE_PATH}/doxy.config.in")
        # failed completely...
        MESSAGE(SEND_ERROR "Please create ${CMAKE_CURRENT_SOURCE_DIR}/doxy.config.in (or doxy.config as fallback)")
      ENDIF(EXISTS "${CMAKE_MODULE_PATH}/doxy.config.in")

    ENDIF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/doxy.config")
  ENDIF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/doxy.config.in")
  
  ADD_CUSTOM_TARGET(doc ${DOXYGEN_EXECUTABLE} ${DOXY_CONFIG})
  
ENDIF(DOXYGEN_FOUND)

ADD_SUBDIRECTORY (specs)
