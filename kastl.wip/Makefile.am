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

AM_CPPFLAGS=@AM_CPPFLAGS@ -I$(top_srcdir)/src -I$(top_srcdir)/api-atha
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

EXTRA_SOURCES=random.cpp kaapi_utils.cpp random.h kaapi_utils.h
transform_SOURCES=      transform.h transform.cpp $(EXTRA_SOURCES)
accumulate_SOURCES=     accumulate.h accumulate.cpp $(EXTRA_SOURCES)
copy_SOURCES=           copy.h copy.cpp $(EXTRA_SOURCES)
count_SOURCES=          count.h count.cpp $(EXTRA_SOURCES)
count_if_SOURCES=       count_if.h count_if.cpp $(EXTRA_SOURCES)
fill_SOURCES=           fill.h fill.cpp $(EXTRA_SOURCES)
find_if_SOURCES=        find_if.h find_if.cpp $(EXTRA_SOURCES)
for_each_SOURCES=       for_each.h for_each.cpp $(EXTRA_SOURCES)
generate_SOURCES=       generate.h generate.cpp $(EXTRA_SOURCES)
inner_product_SOURCES=  inner_product.h inner_product.cpp $(EXTRA_SOURCES)
merge_SOURCES=	        merge.h merge.cpp $(EXTRA_SOURCES)
max_element_SOURCES=    max_element.h max_element.cpp $(EXTRA_SOURCES)
min_element_SOURCES=    min_element.h min_element.cpp $(EXTRA_SOURCES)
replace_copy_SOURCES=   replace_copy.h replace_copy.cpp $(EXTRA_SOURCES)
replace_copy_if_SOURCES=replace_copy_if.h replace_copy_if.cpp $(EXTRA_SOURCES)
replace_SOURCES=        replace.h replace.cpp $(EXTRA_SOURCES)
replace_if_SOURCES=     replace_if.h replace_if.cpp $(EXTRA_SOURCES)
sort_SOURCES=	        sort.h sort.cpp $(EXTRA_SOURCES)
stable_sort_SOURCES=    stable_sort.h stable_sort.cpp $(EXTRA_SOURCES)
swap_ranges_SOURCES=    swap_ranges.h swap_ranges.cpp $(EXTRA_SOURCES)

# put all headers here, to be installed on the system (but random.h and kaapi_utils.h)
# find . -iname "*.h" -printf '\t%P \\\\n' >> Makefile.am   (and edit Makefile.am)
pkginclude_HEADERS=\
	stable_sort.h \
	replace_copy.h \
	replace.h \
	accumulate.h \
	merge.h \
	generate.h \
	replace_copy_if.h \
	count_if.h \
	copy.h \
	sort.h \
	min_element.h \
	swap_ranges.h \
	fill.h \
	for_each.h \
	count.h \
	timing.h \
	replace_if.h \
	transform.h \
	max_element.h \
	find_if.h \
	inner_product.h
