/*
** xkaapi
** 
** Copyright 2008, 2010 INRIA.
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

/* -------------------------------------------------------------------- */
/* Fortran Closure: a fortran closure may have several format, thus
   closure from fortran task stores most of the format information into
   the closure in order to avoid dynamic format of closure creation.
*/
typedef struct FortranTaskArg {
  const kaapi_format_t* task_format; /* description of the task */
} FortranTaskArg;


#if 0/* require next update of XKaapi in order to allows format which depend on task... */
/* Body for all fortran tasks body: dispatch to concret function */
static void fortran_body(void* taskarg, struct kaapi_thread_t* thread )
{
  FortranTaskArg* clo = (FortranTaskArg*)(taskarg);
  switch (clo->_nparam) 
  {
    case 0:
      (*clo->_func)();
      break;
    case 1:
      (*clo->_func)(clo->_param[0]);
      break;
    case 2:
      (*clo->_func)(clo->_param[0], clo->_param[1]);
      break;
    case 3:
      (*clo->_func)(clo->_param[0], clo->_param[1], clo->_param[2]);
      break;
    case 4:
      (*clo->_func)(clo->_param[0], clo->_param[1], clo->_param[2], clo->_param[3]);
      break;
    case 5:
      (*clo->_func)(clo->_param[0], clo->_param[1], clo->_param[2], clo->_param[3], clo->_param[4]);
      break;
    case 6:
      (*clo->_func)(clo->_param[0], clo->_param[1], clo->_param[2], clo->_param[3], clo->_param[4], clo->_param[5]);
      break;
    case 7:
      (*clo->_func)(clo->_param[0], clo->_param[1], clo->_param[2], clo->_param[3], clo->_param[4], clo->_param[5], clo->_param[6]);
      break;
    case 8:
      (*clo->_func)(clo->_param[0], clo->_param[1], clo->_param[2], clo->_param[3], clo->_param[4], clo->_param[5], clo->_param[6], clo->_param[7]);
      break;
    case 9:
      (*clo->_func)(clo->_param[0], clo->_param[1], clo->_param[2], clo->_param[3], clo->_param[4], clo->_param[5], clo->_param[6], clo->_param[7], clo->_param[8]);
      break;
    case 10:
      (*clo->_func)(clo->_param[0], clo->_param[1], clo->_param[2], clo->_param[3], clo->_param[4], clo->_param[5], clo->_param[6], clo->_param[7], clo->_param[8], clo->_param[9]);
      break;
  }
}
#endif


/* -------------------------------------------------------------------- */
/* Declare a new task
   function: address of fortran function
   format: description of the passing mode for each argument
     The expression should be of the form 
       format -> {v type | w | r | x}*
     with type -> {i|f|d|c|z}
     The format describes the passing mode :
       v: by value
       r: shared r
       w: shared w
       x: shared rw
     For the by value passing mode, the type of the argument should be added:
       i: integer
       f: real
       d: double precision
       c: complex
       z: double precision complex
     For instance, 'vixwvd' is the format for 4 arguments of the task : 
       - argument 1 has a passing mode by value ('v') for an integer ('i')
       - argument 2 has a passing mode by reference read-write ('x')
       - argument 3 has a passing mode by reference write ('w')
       - argument 4 has a passing mode by value ('v') for a double precision ('d')
   remainder arguments are in ... and should be passed using the format description.
*/
long kaapi_new_signature_( char* name, void (*function)(), char* format )
{
  kaapi_format_t* formatobject = kaapi_format_allocate();
  
  /* decode format:
     - compute size
     - access mode
     - and offset of each parameter
  */
  size_t size_taskarg =0;
  int count_param = 0;
  int sizearray_param = 32;
  kaapi_access_mode_t* mode_param = malloc( sizeof(kaapi_access_mode_t)*sizearray_param);
  kaapi_offset_t* offset_param = malloc( sizeof(kaapi_offset_t)*sizearray_param);
  const kaapi_format_t** fmt_param = malloc( sizeof(const kaapi_format_t*)*sizearray_param);

  /* count the number of parameter as well as the size of the task args and the offset of each
     parameter.
  */
  kaapi_thread_t bidonthread;
  bidonthread.pc      = 0;
  bidonthread.sp      = 0;
  bidonthread.sp_data = 0;
  char* baseaddr = (char*)kaapi_thread_pushdata( &bidonthread, sizeof(FortranTaskArg) );
  char* c = format;

  while (*c != 0)
  {
    switch (*c) 
    {
      case 'v':
        /* decode the type in order to copy it */
        ++c;
        if ( *c == 0) goto error_label;
        mode_param[count_param] = KAAPI_ACCESS_MODE_V;
        switch (*c) 
        {
          case 'i':
            fmt_param[count_param] = kaapi_int_format;
            offset_param[count_param] = (char*)kaapi_thread_pushdata( &bidonthread, sizeof(int) ) - baseaddr;
          break;

          case 'f':
            fmt_param[count_param] = kaapi_float_format;
            offset_param[count_param] = (char*)kaapi_thread_pushdata( &bidonthread, sizeof(float) ) - baseaddr;
          break;

          case 'd':
            fmt_param[count_param] = kaapi_double_format;
            offset_param[count_param] = (char*)kaapi_thread_pushdata( &bidonthread, sizeof(double) ) - baseaddr;
          break;

          case 'c':
            fmt_param[count_param] = kaapi_complex_format; /* to be define for fortran complex */
            offset_param[count_param] = (char*)kaapi_thread_pushdata( &bidonthread, sizeof(Complex8) ) - baseaddr;
          break;
          
          case 'z':
            fmt_param[count_param] = kaapi_dcomplex_format; /* to be define for fortran complex */
            offset_param[count_param] = (char*)kaapi_thread_pushdata( &bidonthread, sizeof(Complex16) ) - baseaddr;
          break;

          default:
            goto error_label;
        }        
      break;

      case 'r':
        mode_param[count_param]   = KAAPI_ACCESS_MODE_R;
        offset_param[count_param] = (char*)kaapi_thread_pushdata( &bidonthread, sizeof(kaapi_access_t) ) - baseaddr;
      break;

      case 'w':
        mode_param[count_param]   = KAAPI_ACCESS_MODE_W;
        offset_param[count_param] = (char*)kaapi_thread_pushdata( &bidonthread, sizeof(kaapi_access_t) ) - baseaddr;
      break;

      case 'x':
        mode_param[count_param]   = KAAPI_ACCESS_MODE_RW;
        offset_param[count_param] = (char*)kaapi_thread_pushdata( &bidonthread, sizeof(kaapi_access_t) ) - baseaddr;
      break;
      
      default:
        goto error_label;
    }
    ++c;
    ++count_param;
    kaapi_assert( count_param < sizearray_param );
  }
  size_taskarg = (long)bidonthread.sp_data;
  
  kaapi_format_taskregister( formatobject, function, name, 
    size_taskarg,
    count_param,
    mode_param,
    offset_param,
    fmt_param
  );
  return (long)formatobject;

error_label:
  return 0;  
}

void kaapi_new_task_( KAAPI_Fint* addrformat, ... )
{
  const kaapi_format_t* formatobject;
  kaapi_thread_t* thread;
  FortranTaskArg* clo;
  va_list argptr;
  int i;
  void* arg_value;

  typedef KAAPI_Fint* type_arg_i;
  typedef float* type_arg_f;
  typedef double* type_arg_d;
  typedef Complex8* type_arg_c;
  typedef Complex16* type_arg_z;
  typedef kaapi_access_t* type_arg_access;
  
  if (addrformat ==0)
    goto error_label;

  formatobject = (const kaapi_format_t*)addrformat;
  thread = kaapi_self_thread();

  clo = (FortranTaskArg*)kaapi_thread_pushdata(thread, sizeof(FortranTaskArg) );
  clo->task_format = formatobject;

  va_start( argptr, addrformat );
  for (i=0; i<formatobject->count_params; ++i)
  {
    if (formatobject->mode_params[i] == KAAPI_ACCESS_MODE_V)
    {
      arg_value = va_arg( argptr, void* );
      if (formatobject->fmt_params[i] == kaapi_int_format)
      {
        type_arg_i pf = (type_arg_i)kaapi_thread_pushdata( thread, sizeof(int) );
        kaapi_assert_debug( formatobject->off_params[i] == (long)pf - (long)clo );
        *pf = *(type_arg_i)arg_value;
      }
      else if (formatobject->fmt_params[i] == kaapi_float_format)
      {
        type_arg_f pf = (type_arg_f)kaapi_thread_pushdata( thread, sizeof(float) );
        kaapi_assert_debug( formatobject->off_params[i] == (long)pf - (long)clo );
        *pf = *(type_arg_f)arg_value;
      }
      else if (formatobject->fmt_params[i] == kaapi_double_format)
      {
        type_arg_f pf = (type_arg_f)kaapi_thread_pushdata( thread, sizeof(double) );
        kaapi_assert_debug( formatobject->off_params[i] == (long)pf - (long)clo );
        *pf = *(type_arg_f)arg_value;
      }
      else if (formatobject->fmt_params[i] == kaapi_complex_format)
      {
        type_arg_c pf = (type_arg_c)kaapi_thread_pushdata( thread, sizeof(double) );
        kaapi_assert_debug( formatobject->off_params[i] == (long)pf - (long)clo );
        *pf = *(type_arg_c)arg_value;
      }
      else if (formatobject->fmt_params[i] == kaapi_dcomplex_format)
      {
        type_arg_z pf = (type_arg_z)kaapi_thread_pushdata( thread, sizeof(double) );
        kaapi_assert_debug( formatobject->off_params[i] == (long)pf - (long)clo );
        *pf = *(type_arg_z)arg_value;
      }
      else goto error_label;
    }
    else { /* a shared memory reference, arg is passed by value */
      type_arg_access pf = (type_arg_access)kaapi_thread_pushdata(thread, sizeof(kaapi_access_t) );
      kaapi_assert_debug( formatobject->off_params[i] == (long)pf - (long)clo );
      arg_value = va_arg( argptr, void* );
      kaapi_access_init(pf, arg_value);
    }
  }
  va_end( argptr );
    
  /* push closure */
  kaapi_task_init(kaapi_thread_toptask(thread), formatobject->entrypoint[KAAPI_PROC_TYPE_CPU], clo);
  kaapi_thread_pushtask(thread);
  return;

error_label:
  return;  
}


/* -------------------------------------------------------------------- */
void kaapi_sync_()
{
  kaapi_sched_sync();
}

