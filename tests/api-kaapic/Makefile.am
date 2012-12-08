##
##  xkaapi
##
##  Created by CL and TG on 04/02/09.
##  Copyright 2009,2010,2011,2012 INRIA. All rights reserved.
##

AM_CPPFLAGS=@AM_CPPFLAGS@ -Werror $(KAAPICPROGS_BUILD_CPPFLAGS)
AM_CFLAGS=@AM_CFLAGS@ $(STD_C99_FLAGS) $(KAAPICPROGS_BUILD_CFLAGS)
AM_CXXFLAGS=@AM_CXXFLAGS@ $(STD_CXX0X_FLAGS) $(KAAPICPROGS_BUILD_CXXFLAGS)
AM_LDFLAGS=@AM_LDFLAGS@ $(KAAPICPROGS_BUILD_LDFLAGS)
AM_LIBS=@AM_LIBS@ $(KAAPICPROGS_BUILD_LIBS)

check_headers_c_headers_SRC=  check-headers/c/kaapic.h.c
check_headers_cxx_headers_SRC=check-headers/c++/kaapic.h.c++
check_headers_cppflags=@AM_CPPFLAGS@ $(KAAPICPROGS_BUILD_CPPFLAGS)
check_headers_cflags=  @AM_CFLAGS@   $(KAAPICPROGS_BUILD_CFLAGS)
check_headers_cxxflags=@AM_CXXFLAGS@ $(KAAPICPROGS_BUILD_CXXFLAGS)
check_headers_ldflags= @AM_LDFLAGS@  $(KAAPICPROGS_BUILD_LDFLAGS)
check_headers_libs=                  $(KAAPICPROGS_BUILD_LIBS)

if  KAAPI_API_KAAPIC

include ../check-headers/check-headers.am

noinst_PROGRAMS+=   \
  $(check_headers_prgs) \
  test_begend_parallel \
  spawn_s-arg \
  spawn_s5-arg \
  spawn_v-arg_char \
  spawn_v-arg_schar \
  spawn_v-arg_shrt \
  spawn_v-arg_int \
  spawn_v-arg_long \
  spawn_v-arg_llong \
  spawn_v-arg_int8 \
  spawn_v-arg_int16 \
  spawn_v-arg_int32 \
  spawn_v-arg_int64 \
  spawn_v-arg_uchar \
  spawn_v-arg_ushrt \
  spawn_v-arg_uint \
  spawn_v-arg_ulong \
  spawn_v-arg_ullong \
  spawn_v-arg_uint8 \
  spawn_v-arg_uint16 \
  spawn_v-arg_uint32 \
  spawn_v-arg_uint64 \
  spawn_v-arg_flt \
  spawn_v-arg_dbl \
  spawn_v-arg_ldbl \
  spawn_v-arg_pointer \
  nqueens_stackpointer\
  fibo_sync \
  fibo_sum \
  fibo_cumul
endif

test_begend_parallel_SOURCES=test_begend_parallel.c
fibo_sync_SOURCES=fibo_sync.c fib_verify.h
fibo_sum_SOURCES=fibo_sum.c fib_verify.h
fibo_cumul_SOURCES=fibo_cumul.c fib_verify.h
nqueens_stackpointer_SOURCES=nqueens_stackpointer.c
spawn_s_arg_SOURCES=spawn_s-arg.c
spawn_s5_arg_SOURCES=spawn_s5-arg.c
spawn_v_arg_char_SOURCES=spawn_v-arg.c
spawn_v_arg_char_CPPFLAGS=$(AM_CPPFLAGS) -DT="char" -DKT="CHAR"
spawn_v_arg_schar_SOURCES=spawn_v-arg.c
spawn_v_arg_schar_CPPFLAGS=$(AM_CPPFLAGS) -DT="signed char" -DKT="SCHAR"
spawn_v_arg_shrt_SOURCES=spawn_v-arg.c
spawn_v_arg_shrt_CPPFLAGS=$(AM_CPPFLAGS) -DT="short" -DKT="SHRT"
spawn_v_arg_int_SOURCES=spawn_v-arg.c
spawn_v_arg_int_CPPFLAGS=$(AM_CPPFLAGS) -DT="int" -DKT="INT"
spawn_v_arg_long_SOURCES=spawn_v-arg.c
spawn_v_arg_long_CPPFLAGS=$(AM_CPPFLAGS) -DT="long" -DKT="LONG"
spawn_v_arg_llong_SOURCES=spawn_v-arg.c
spawn_v_arg_llong_CPPFLAGS=$(AM_CPPFLAGS) -DT="long long" -DKT="LLONG"
spawn_v_arg_int8_SOURCES=spawn_v-arg.c
spawn_v_arg_int8_CPPFLAGS=$(AM_CPPFLAGS) -DT="int8_t" -DKT="INT8"
spawn_v_arg_int16_SOURCES=spawn_v-arg.c
spawn_v_arg_int16_CPPFLAGS=$(AM_CPPFLAGS) -DT="int16_t" -DKT="INT16"
spawn_v_arg_int32_SOURCES=spawn_v-arg.c
spawn_v_arg_int32_CPPFLAGS=$(AM_CPPFLAGS) -DT="int32_t" -DKT="INT32"
spawn_v_arg_int64_SOURCES=spawn_v-arg.c
spawn_v_arg_int64_CPPFLAGS=$(AM_CPPFLAGS) -DT="int64_t" -DKT="INT64"
spawn_v_arg_uchar_SOURCES=spawn_v-arg.c
spawn_v_arg_uchar_CPPFLAGS=$(AM_CPPFLAGS) -DT="unsigned char" -DKT="UCHAR"
spawn_v_arg_ushrt_SOURCES=spawn_v-arg.c
spawn_v_arg_ushrt_CPPFLAGS=$(AM_CPPFLAGS) -DT="unsigned short" -DKT="USHRT"
spawn_v_arg_uint_SOURCES=spawn_v-arg.c
spawn_v_arg_uint_CPPFLAGS=$(AM_CPPFLAGS) -DT="unsigned int" -DKT="UINT"
spawn_v_arg_ulong_SOURCES=spawn_v-arg.c
spawn_v_arg_ulong_CPPFLAGS=$(AM_CPPFLAGS) -DT="unsigned long" -DKT="ULONG"
spawn_v_arg_ullong_SOURCES=spawn_v-arg.c
spawn_v_arg_ullong_CPPFLAGS=$(AM_CPPFLAGS) -DT="unsigned long long" -DKT="ULLONG"
spawn_v_arg_uint8_SOURCES=spawn_v-arg.c
spawn_v_arg_uint8_CPPFLAGS=$(AM_CPPFLAGS) -DT="uint8_t" -DKT="UINT8"
spawn_v_arg_uint16_SOURCES=spawn_v-arg.c
spawn_v_arg_uint16_CPPFLAGS=$(AM_CPPFLAGS) -DT="uint16_t" -DKT="UINT16"
spawn_v_arg_uint32_SOURCES=spawn_v-arg.c
spawn_v_arg_uint32_CPPFLAGS=$(AM_CPPFLAGS) -DT="uint32_t" -DKT="UINT32"
spawn_v_arg_uint64_SOURCES=spawn_v-arg.c
spawn_v_arg_uint64_CPPFLAGS=$(AM_CPPFLAGS) -DT="uint64_t" -DKT="UINT64"
spawn_v_arg_flt_SOURCES=spawn_v-arg.c
spawn_v_arg_flt_CPPFLAGS=$(AM_CPPFLAGS) -DT="float" -DKT="FLT"
spawn_v_arg_dbl_SOURCES=spawn_v-arg.c
spawn_v_arg_dbl_CPPFLAGS=$(AM_CPPFLAGS) -DT="double" -DKT="DBL"
spawn_v_arg_ldbl_SOURCES=spawn_v-arg.c
spawn_v_arg_ldbl_CPPFLAGS=$(AM_CPPFLAGS) -DT="long double" -DKT="LDBL"
spawn_v_arg_pointer_SOURCES=spawn_v-arg.c
spawn_v_arg_pointer_CPPFLAGS=$(AM_CPPFLAGS) -DT="void*" -DKT="PTR"

