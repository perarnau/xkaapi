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
#include <stdlib.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <unistd.h>

/*
*/
kaapi_rtparam_t default_param;

/** Predefined format
*/
kaapi_format_t kaapi_shared_format;

kaapi_format_t kaapi_char_format;
kaapi_format_t kaapi_short_format;
kaapi_format_t kaapi_int_format;
kaapi_format_t kaapi_long_format;
kaapi_format_t kaapi_longlong_format;

kaapi_format_t kaapi_uchar_format;
kaapi_format_t kaapi_ushort_format;
kaapi_format_t kaapi_uint_format;
kaapi_format_t kaapi_ulong_format;
kaapi_format_t kaapi_ulonglong_format;

kaapi_format_t kaapi_float_format;
kaapi_format_t kaapi_double_format;


/**
*/
extern int kaapi_setup_param( int argc, char** argv )
{
  /* compute the number of cpu of the system */
#if defined(KAAPI_USE_LINUX)
  default_param.syscpucount = sysconf(_SC_NPROCESSORS_CONF);
#elif defined(KAAPI_USE_APPLE)
  {
    int mib[2];
    size_t len;
    mib[0] = CTL_HW;
    mib[1] = HW_NCPU;
    len = sizeof(default_param.syscpucount);
    sysctl(mib, 2, &default_param.syscpucount, &len, 0, 0);
  }
#else
  #warning "Could not compute number of physical cpu of the system. Default value==1"
  default_param.syscpucount = 1;
#endif
  /* adjust system limit, if library is compiled with greather number of processors that available */
  if (default_param.syscpucount < KAAPI_MAX_PROCESSOR)
    default_param.syscpucount = KAAPI_MAX_PROCESSOR;
    
  /* Set default values */
  default_param.cpucount  = default_param.syscpucount;
  default_param.stacksize = 8*4096;
  
  /* Get values from environment variable */
  if (getenv("KAAPI_STACKSIZE") !=0)
  {
    default_param.stacksize = atoi(getenv("KAAPI_STACKSIZE"));
  }
  if (getenv("KAAPI_CPUCOUNT") !=0)
  {
    default_param.cpucount = atoi(getenv("KAAPI_CPUCOUNT"));
  }
  
  /* default workstealing selection function */
  default_param.wsselect = &kaapi_sched_select_victim_rand;

  /* TODO: here parse command line option */
  
  return 0;
}



/**
*/
#define KAAPI_REGISTER_BASICTYPEFORMAT( formatobject, type ) \
  static void formatobject##_cstor(void* dest)  { *(type*)dest = 0; }\
  static void formatobject##_dstor(void* dest) { *(type*)dest = 0; }\
  static void formatobject##_cstorcopy( void* dest, const void* src) { *(type*)dest = *(type*)src; } \
  static void formatobject##_copy( void* dest, const void* src) { *(type*)dest = *(type*)src; } \
  static void formatobject##_assign( void* dest, const void* src) { *(type*)dest = *(type*)src; } \
  static inline kaapi_format_t* fnc_##formatobject(void) \
  {\
    return &formatobject;\
  }\
  static inline void __attribute__ ((constructor)) __kaapi_register_format_##formatobject (void)\
  { \
    static int isinit = 0;\
    if (isinit) return;\
    isinit = 1;\
    kaapi_format_structregister( &fnc_##formatobject, \
                                 #type, sizeof(type), \
                                 &formatobject##_cstor, &formatobject##_dstor, &formatobject##_cstorcopy, \
                                 &formatobject##_copy, &formatobject##_assign ); \
  }


KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_char_format, char)
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_short_format, short)
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_int_format, int)
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_long_format, long)
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_longlong_format, long long)


KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_uchar_format, unsigned char)
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_ushort_format, unsigned short)
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_uint_format, unsigned int)
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_ulong_format, unsigned long)
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_ulonglong_format, unsigned long long)

KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_float_format, float)
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_double_format, double)


