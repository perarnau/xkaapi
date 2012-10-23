/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
**
** Contributors :
**
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
#include <string.h>


/**
*/
#define KAAPI_DECL_BASICTYPEFORMAT( formatobject, type, fmt ) \
  kaapi_format_t formatobject##_object;\
  kaapi_format_t* formatobject= &formatobject##_object;\
  static void formatobject##_cstor(void* dest)  { *(type*)dest = 0; }\
  static void formatobject##_dstor(void* dest) { *(type*)dest = 0; }\
  static void formatobject##_cstorcopy( void* dest, const void* src) { *(type*)dest = *(type*)src; } \
  static void formatobject##_copy( void* dest, const void* src) { *(type*)dest = *(type*)src; } \
  static void formatobject##_assign( void* dest, const kaapi_memory_view_t* dview, const void* src, const kaapi_memory_view_t* sview) { *(type*)dest = *(type*)src; } \
  static void formatobject##_print( FILE* file, const void* src) { fprintf(file, fmt, *(type*)src); } \
  kaapi_format_t* get_##formatobject(void) \
  {\
    return &formatobject##_object;\
  }

#define KAAPI_REGISTER_BASICTYPEFORMAT( formatobject, type, fmt ) \
  kaapi_format_structregister( &get_##formatobject, \
                               #type, sizeof(type), \
                               &formatobject##_cstor, &formatobject##_dstor, &formatobject##_cstorcopy, \
                               &formatobject##_copy, &formatobject##_assign, &formatobject##_print ); \



/** Predefined format
*/
KAAPI_DECL_BASICTYPEFORMAT(kaapi_char_format, char, "%hhu")
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
KAAPI_DECL_BASICTYPEFORMAT(kaapi_longdouble_format, long double, "%Le")  

/* void pointer format */
static void voidp_type_cstor(void* addr)
{
  /* printf("%s\n", __FUNCTION__); */
  *(void**)addr = 0;
}

static void voidp_type_dstor(void* addr)
{
  /* printf("%s\n", __FUNCTION__); */
  *(void**)addr = 0;
}

static void voidp_type_cstorcopy(void* daddr, const void* saddr)
{
  /* TODO: missing views */
  /* printf("%s\n", __FUNCTION__); */
}

static void voidp_type_copy(void* daddr, const void* saddr)
{
  /* TODO: missing views */
  /* printf("%s\n", __FUNCTION__); */
}

static void voidp_type_assign
(
 void* daddr, const kaapi_memory_view_t* dview,
 const void* saddr, const kaapi_memory_view_t* sview
)
{
  memcpy(daddr, saddr, kaapi_memory_view_size(dview));
}

static void voidp_type_printf(FILE* fil, const void* addr)
{
  fprintf(fil, "0x%lx", (uintptr_t)addr);
}

kaapi_format_t kaapi_voidp_format_object;
kaapi_format_t* get_kaapi_voidp_format(void)
{
  /* printf("%s\n", __FUNCTION__); */
  return &kaapi_voidp_format_object;
}

kaapi_format_t* kaapi_voidp_format = &kaapi_voidp_format_object;



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
  KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_longdouble_format, long double, "%le")  

  kaapi_format_structregister
  ( 
   get_kaapi_voidp_format,
   "kaapi_voidp_format",
   sizeof(void*),
   voidp_type_cstor,
   voidp_type_dstor,
   voidp_type_cstorcopy,
   voidp_type_copy,
   voidp_type_assign,
   voidp_type_printf
  );
  
  /* register format for some internal tasks */
  kaapi_register_staticschedtask_format();
}
