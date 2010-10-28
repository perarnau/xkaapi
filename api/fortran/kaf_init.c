/*
** xkaapi
** 
** Copyright 2007, 2010 INRIA.
**
** Contributors :
**
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
#include "kaf_impl.h"
#include <stdarg.h>
#include <stdio.h>

/* API for Fortran */

extern int  get_fortran_argc_();
extern void get_fortran_argv_(int *i, char* s);

#define KAAPIF_DECL_BASICTYPEFORMAT( formatobject, type, fmt ) \
  kaapi_format_t formatobject##_object;\
  kaapi_format_t* formatobject= &formatobject##_object;\
  static void formatobject##_cstor(void* dest)  { memset(dest, 0, sizeof(type)); }\
  static void formatobject##_dstor(void* dest) { memset(dest, 0, sizeof(type)); }\
  static void formatobject##_cstorcopy( void* dest, const void* src) { memcpy(dest, src, sizeof(type)); } \
  static void formatobject##_copy( void* dest, const void* src) { memcpy(dest, src, sizeof(type)); } \
  static void formatobject##_assign( void* dest, const void* src) { memcpy(dest, src, sizeof(type));} \
  static void formatobject##_print( FILE* file, const void* src) { fprintf(file, fmt, ((type*)src)->r, ((type*)src)->i); } \
  static kaapi_format_t* fnc_##formatobject(void) \
  {\
    return &formatobject##_object;\
  }

#define KAAPIF_REGISTER_BASICTYPEFORMAT( formatobject, type, fmt ) \
  kaapi_format_structregister( &fnc_##formatobject, \
                               #type, sizeof(type), \
                               &formatobject##_cstor, &formatobject##_dstor, &formatobject##_cstorcopy, \
                               &formatobject##_copy, &formatobject##_assign, &formatobject##_print ); \

KAAPIF_DECL_BASICTYPEFORMAT(kaapi_complex_format, Complex8, "%e,%e")
KAAPIF_DECL_BASICTYPEFORMAT(kaapi_dcomplex_format, Complex16, "%e,%e")


/* -------------------------------------------------------------------- */
void FNAME(kaapi_init)( KAAPI_Fint* ierr )
{ 
  /* Here is for gfortran that export these variables
  */
#if defined(KAAPI_DEBUG)  
  int i;
  printf("[KFortran] Argc=%i\n", get_fortran_argc_() );

  for (i=0; i<get_fortran_argc_(); ++i)
  {
    //warning: everything static here, bugs if size of args > 256
    //changes in size need to be propagated in corresponding fortran file
    //and below in this file
    char tmp[256];
    get_fortran_argv_(&i, tmp);
    printf("[KFortran] Argv[%i]=%s\n", i, tmp );
  }
#endif
  KAAPIF_REGISTER_BASICTYPEFORMAT(kaapi_complex_format, Complex8, "%e,%e")
  KAAPIF_REGISTER_BASICTYPEFORMAT(kaapi_dcomplex_format, Complex16, "%e,%e")
  *ierr = 0;
}


/* -------------------------------------------------------------------- */
/* Initialize the library
*/
void FNAME(kaapi_finalize)( KAAPI_Fint* ierr )
{
  kaapi_sched_sync();
  *ierr = 0;
}


/* -------------------------------------------------------------------- */
/* Initialize the library
*/
int FNAME(kaapi_isleader)(void)
{
  return 1;
}
