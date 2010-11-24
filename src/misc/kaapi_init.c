/*
** kaapi_init.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:03 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
** 
** This software is a computer program whose purpose is to execute
** multithreaded computation with data flow synchronization between
** threads.
** 
** This software is governed by the CeCILL-C license under French law
** and abiding by the rules of distribution of free software.  You can
** use, modify and/ or redistribute the software under the terms of
** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
** following URL "http://www.cecill.info".
** 
** As a counterpart to the access to the source code and rights to
** copy, modify and redistribute granted by the license, users are
** provided only with a limited warranty and the software's author,
** the holder of the economic rights, and the successive licensors
** have only limited liability.
** 
** In this respect, the user's attention is drawn to the risks
** associated with loading, using, modifying and/or developing or
** reproducing the software by the user in light of its specific
** status of free software, that may mean that it is complicated to
** manipulate, and that also therefore means that it is reserved for
** developers and experienced professionals having in-depth computer
** knowledge. Users are therefore encouraged to load and test the
** software's suitability as regards their requirements in conditions
** enabling the security of their systems and/or data to be ensured
** and, more generally, to use and operate it in the same conditions
** as regards security.
** 
** The fact that you are presently reading this means that you have
** had knowledge of the CeCILL-C license and that you accept its
** terms.
** 
*/
#include "kaapi_impl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#if defined (_WIN32) || defined(KAAPI_USE_APPLE)
#include <sys/sysctl.h>
#endif
#include <unistd.h>
#include <errno.h>


/*
*/
kaapi_rtparam_t kaapi_default_param = {
   .startuptime = 0,
   .stacksize   = 64*4096,
   .cpucount    = 0,
};


/**
*/
#define KAAPI_DECL_BASICTYPEFORMAT( formatobject, type, fmt ) \
  kaapi_format_t formatobject##_object;\
  kaapi_format_t* formatobject= &formatobject##_object;\
  static void formatobject##_cstor(void* dest)  { *(type*)dest = 0; }\
  static void formatobject##_dstor(void* dest) { *(type*)dest = 0; }\
  static void formatobject##_cstorcopy( void* dest, const void* src) { *(type*)dest = *(type*)src; } \
  static void formatobject##_copy( void* dest, const void* src) { *(type*)dest = *(type*)src; } \
  static void formatobject##_assign( void* dest, const void* src) { *(type*)dest = *(type*)src; } \
  static void formatobject##_print( FILE* file, const void* src) { fprintf(file, fmt, *(type*)src); } \
  static kaapi_format_t* fnc_##formatobject(void) \
  {\
    return &formatobject##_object;\
  }

#define KAAPI_REGISTER_BASICTYPEFORMAT( formatobject, type, fmt ) \
  kaapi_format_structregister( &fnc_##formatobject, \
                               #type, sizeof(type), \
                               &formatobject##_cstor, &formatobject##_dstor, &formatobject##_cstorcopy, \
                               &formatobject##_copy, &formatobject##_assign, &formatobject##_print ); \



/** Predefined format
*/
KAAPI_DECL_BASICTYPEFORMAT(kaapi_char_format, char, "%hhi")
KAAPI_DECL_BASICTYPEFORMAT(kaapi_short_format, short, "%hi")
KAAPI_DECL_BASICTYPEFORMAT(kaapi_int_format, int, "%i")
KAAPI_DECL_BASICTYPEFORMAT(kaapi_long_format, long, "%li")
KAAPI_DECL_BASICTYPEFORMAT(kaapi_longlong_format, long long, "%lli")
KAAPI_DECL_BASICTYPEFORMAT(kaapi_uchar_format, unsigned char, "%hhu")
KAAPI_DECL_BASICTYPEFORMAT(kaapi_ushort_format, unsigned short, "%hu")
KAAPI_DECL_BASICTYPEFORMAT(kaapi_uint_format, unsigned int, "%u")
KAAPI_DECL_BASICTYPEFORMAT(kaapi_ulong_format, unsigned long, "%lu")
KAAPI_DECL_BASICTYPEFORMAT(kaapi_ulonglong_format, unsigned long long, "%llu")
KAAPI_DECL_BASICTYPEFORMAT(kaapi_float_format, float, "%e")
KAAPI_DECL_BASICTYPEFORMAT(kaapi_double_format, double, "%e")  


/** \ingroup WS
    Initialize from xkaapi runtime parameters from command line
    \param argc [IN] command line argument count
    \param argv [IN] command line argument vector
    \retval 0 in case of success 
    \retval EINVAL because of error when parsing then KAAPI_CPUSET string
    \retval E2BIG because of a cpu index too high in KAAPI_CPUSET
    
*/
int kaapi_setup_param( 
  int argc __attribute__((unused)), 
  char** argv __attribute__((unused))
)
{
  const char* wsselect;
    
  /* compute the number of cpu of the system */
#if defined(KAAPI_USE_LINUX)
  kaapi_default_param.syscpucount = sysconf(_SC_NPROCESSORS_CONF);
#elif defined(KAAPI_USE_APPLE)
  {
    int mib[2];
    size_t len;
    mib[0] = CTL_HW;
    mib[1] = HW_NCPU;
    len = sizeof(kaapi_default_param.syscpucount);
    sysctl(mib, 2, &kaapi_default_param.syscpucount, &len, 0, 0);
  }
#elif defined(_WIN32)
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  kaapi_default_param.syscpucount = sys_info.dwNumberOfProcessors;
#else
  #warning "Could not compute number of physical cpu of the system. Default value==1"
  kaapi_default_param.syscpucount = 1;
#endif
  /* adjust system limit, if library is compiled with greather number of processors that available */
  if (kaapi_default_param.syscpucount > KAAPI_MAX_PROCESSOR)
    kaapi_default_param.syscpucount = KAAPI_MAX_PROCESSOR;

  kaapi_default_param.use_affinity = 0;

  kaapi_default_param.cpucount  = kaapi_default_param.syscpucount;
  
  if (getenv("KAAPI_DISPLAY_PERF") !=0)
    kaapi_default_param.display_perfcounter = 1;
  else
    kaapi_default_param.display_perfcounter = 0;

  if (getenv("KAAPI_STACKSIZE") !=0)
    kaapi_default_param.stacksize = atoi(getenv("KAAPI_STACKSIZE"));

  /* workstealing selection function */
  wsselect = getenv("KAAPI_WSSELECT");
  kaapi_default_param.wsselect = &kaapi_sched_select_victim_rand;
  if ((wsselect != NULL) && (strcmp(wsselect, "workload") ==0))
    kaapi_default_param.wsselect = &kaapi_sched_select_victim_workload_rand;
  else if ((wsselect != NULL) && (strcmp(wsselect, "first0") ==0))
    kaapi_default_param.wsselect = &kaapi_sched_select_victim_workload_rand;
  
  
  return 0;
}


/**
*/
void kaapi_init_basicformat(void)
{
  KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_char_format, char, "%hhi")
  KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_short_format, short, "%hi")
  KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_int_format, int, "%i")
  KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_long_format, long, "%li")
  KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_longlong_format, long long, "%lli")
  KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_uchar_format, unsigned char, "%hhu")
  KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_ushort_format, unsigned short, "%hu")
  KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_uint_format, unsigned int, "%u")
  KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_ulong_format, unsigned long, "%lu")
  KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_ulonglong_format, unsigned long long, "%llu")
  KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_float_format, float, "%e")
  KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_double_format, double, "%e")  
}


/**
*/
int kaapi_init(void)
{
  static int iscalled = 0;
  if (iscalled !=0) return EALREADY;
  iscalled = 1;

  kaapi_init_basicformat();
  
  /* set up runtime parameters */
  kaapi_assert_m( 0 ==kaapi_setup_param( 0, 0 ), "kaapi_setup_param" );
  
  int err = kaapi_mt_init();
  return err;
}


/**
*/
int kaapi_finalize(void)
{
  return kaapi_mt_finalize();
}
