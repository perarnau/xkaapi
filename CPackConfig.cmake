# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. Example variables are:
#   CPACK_GENERATOR                     - Generator used to create package
#   CPACK_INSTALL_CMAKE_PROJECTS        - For each project (path, name, component)
#   CPACK_CMAKE_GENERATOR               - CMake Generator used for the projects
#   CPACK_INSTALL_COMMANDS              - Extra commands to install components
#   CPACK_INSTALL_DIRECTORIES           - Extra directories to install
#   CPACK_PACKAGE_DESCRIPTION_FILE      - Description file for the package
#   CPACK_PACKAGE_DESCRIPTION_SUMMARY   - Summary of the package
#   CPACK_PACKAGE_EXECUTABLES           - List of pairs of executables and labels
#   CPACK_PACKAGE_FILE_NAME             - Name of the package generated
#   CPACK_PACKAGE_ICON                  - Icon used for the package
#   CPACK_PACKAGE_INSTALL_DIRECTORY     - Name of directory for the installer
#   CPACK_PACKAGE_NAME                  - Package project name
#   CPACK_PACKAGE_VENDOR                - Package project vendor
#   CPACK_PACKAGE_VERSION               - Package project version
#   CPACK_PACKAGE_VERSION_MAJOR         - Package project version (major)
#   CPACK_PACKAGE_VERSION_MINOR         - Package project version (minor)
#   CPACK_PACKAGE_VERSION_PATCH         - Package project version (patch)

# There are certain generator specific ones

# NSIS Generator:
#   CPACK_PACKAGE_INSTALL_REGISTRY_KEY  - Name of the registry key for the installer
#   CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS - Extra commands used during uninstall
#   CPACK_NSIS_EXTRA_INSTALL_COMMANDS   - Extra commands used during install


SET(CPACK_BINARY_BUNDLE "")
SET(CPACK_BINARY_CYGWIN "")
SET(CPACK_BINARY_DEB "ON")
SET(CPACK_BINARY_NSIS "")
SET(CPACK_BINARY_OSXX11 "OFF")
SET(CPACK_BINARY_PACKAGEMAKER "ON")
SET(CPACK_BINARY_RPM "OFF")
SET(CPACK_BINARY_STGZ "ON")
SET(CPACK_BINARY_TBZ2 "OFF")
SET(CPACK_BINARY_TGZ "ON")
SET(CPACK_BINARY_TZ "")
SET(CPACK_BINARY_ZIP "")
SET(CPACK_CMAKE_GENERATOR "Unix Makefiles")
SET(CPACK_COMPONENTS_ALL "")
SET(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
SET(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
SET(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.3.1-6), libgcc1 (>= 1:3.4.2-12)")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "christophe laferriere")
SET(CPACK_DEBIAN_PACKAGE_NAME "libxkaapi-dev")
SET(CPACK_GENERATOR "PackageMaker;DEB;STGZ;TGZ")
SET(CPACK_INSTALL_CMAKE_PROJECTS "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi;CKAAPI;ALL;/")
SET(CPACK_INSTALL_PREFIX "/usr/local")
SET(CPACK_MODULE_PATH "")
SET(CPACK_NSIS_DISPLAY_NAME "libxkaapi 1.0.0")
SET(CPACK_NSIS_INSTALLER_ICON_CODE "")
SET(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
SET(CPACK_OUTPUT_CONFIG_FILE "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/CPackConfig.cmake")
SET(CPACK_PACKAGE_DEFAULT_LOCATION "/")
SET(CPACK_PACKAGE_DESCRIPTION_FILE "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/README.txt")
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "X-Kaapi is a C library that allows to
  execute multithreaded computation with data flow synchronization between
  threads")
SET(CPACK_PACKAGE_FILE_NAME "libxkaapi-1.0.0-Darwin")
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "libxkaapi 1.0.0")
SET(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "libxkaapi 1.0.0")
SET(CPACK_PACKAGE_NAME "libxkaapi")
SET(CPACK_PACKAGE_RELOCATABLE "true")
SET(CPACK_PACKAGE_VENDOR "INRIA")
SET(CPACK_PACKAGE_VERSION "1.0.0")
SET(CPACK_PACKAGE_VERSION_MAJOR "1")
SET(CPACK_PACKAGE_VERSION_MINOR "0")
SET(CPACK_PACKAGE_VERSION_PATCH "0")
SET(CPACK_RESOURCE_FILE_LICENSE "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/LICENCE.txt")
SET(CPACK_RESOURCE_FILE_README "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/README.txt")
SET(CPACK_RESOURCE_FILE_WELCOME "/Applications/CMake 2.6-2.app/Contents/share/cmake-2.6/Templates/CPack.GenericWelcome.txt")
SET(CPACK_SET_DESTDIR "OFF")
SET(CPACK_SOURCE_CYGWIN "")
SET(CPACK_SOURCE_GENERATOR "TGZ")
SET(CPACK_SOURCE_IGNORE_FILES "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/.git;/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/.gitignore;")
SET(CPACK_SOURCE_OUTPUT_CONFIG_FILE "/Users/thierry/Developpment/KAAPI/kaapi/git/xkaapi/xkaapi/CPackSourceConfig.cmake")
SET(CPACK_SOURCE_TBZ2 "")
SET(CPACK_SOURCE_TGZ "")
SET(CPACK_SOURCE_TZ "")
SET(CPACK_SOURCE_ZIP "")
SET(CPACK_SYSTEM_NAME "Darwin")
SET(CPACK_TOPLEVEL_TAG "Darwin")
