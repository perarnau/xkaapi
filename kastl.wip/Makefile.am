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

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(top_srcdir)/src/machine/$(KAAPI_MACHINE) -I$(top_srcdir)/api-atha
AM_CXXFLAGS=@AM_CXXFLAGS@ $(ATHA1PROGS_BUILD_CXXFLAGS) -std=c++0x
AM_LDFLAGS =@AM_LDFLAGS@  $(ATHA1PROGS_BUILD_LDFLAGS)
LIBS       =@LIBS@        $(ATHA1PROGS_BUILD_LIBS)     ../api-atha/libatha1.la ../src/libxkaapi.la


# NEW TARGET Executable Section :
check_PROGRAMS=\
	transform     	      \
	accumulate     	      \
	copy     	      \
	count     	      \
	count_if     	      \
	fill     	      \
	find_if     	      \
	for_each     	      \
	generate     	      \
	inner_product 	      \
	merge		      \
	max_element 	      \
	min_element 	      \
	replace_copy 	      \
	replace_copy_if	      \
	replace		      \
	replace_if	      \
	sort		      \
	stable_sort	      \
	swap_ranges

transform_SOURCES=      transform.h transform.cpp random.cpp kaapi_utils.cpp
accumulate_SOURCES=     accumulate.h accumulate.cpp random.cpp kaapi_utils.cpp
copy_SOURCES=           copy.h copy.cpp random.cpp kaapi_utils.cpp
count_SOURCES=          count.h count.cpp random.cpp kaapi_utils.cpp
count_if_SOURCES=       count_if.h count_if.cpp random.cpp kaapi_utils.cpp
fill_SOURCES=           fill.h fill.cpp random.cpp kaapi_utils.cpp
find_if_SOURCES=        find_if.h find_if.cpp random.cpp kaapi_utils.cpp
for_each_SOURCES=       for_each.h for_each.cpp random.cpp kaapi_utils.cpp
generate_SOURCES=       generate.h generate.cpp random.cpp kaapi_utils.cpp
inner_product_SOURCES=  inner_product.h inner_product.cpp random.cpp kaapi_utils.cpp
merge_SOURCES=	        merge.h merge.cpp random.cpp kaapi_utils.cpp
max_element_SOURCES=    max_element.h max_element.cpp random.cpp kaapi_utils.cpp
min_element_SOURCES=    min_element.h min_element.cpp random.cpp kaapi_utils.cpp
replace_copy_SOURCES=   replace_copy.h replace_copy.cpp random.cpp kaapi_utils.cpp
replace_copy_if_SOURCES=replace_copy_if.h replace_copy_if.cpp random.cpp kaapi_utils.cpp
replace_SOURCES=        replace.h replace.cpp random.cpp kaapi_utils.cpp
replace_if_SOURCES=     replace_if.h replace_if.cpp random.cpp kaapi_utils.cpp
sort_SOURCES=	        sort.h sort.cpp random.cpp kaapi_utils.cpp
stable_sort_SOURCES=    stable_sort.h stable_sort.cpp random.cpp kaapi_utils.cpp
swap_ranges_SOURCES=    swap_ranges.h swap_ranges.cpp random.cpp kaapi_utils.cpp

