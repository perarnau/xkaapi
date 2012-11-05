/*
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:19:14 2009
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributors :
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

#include "kaapif.h"
#include "kaapic_impl.h"
#include "kaapic_dfg_switch.h"
#include <string.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>


/* dataflow interface */
int kaapif_spawn_
(
 int32_t* nargs,
 void (*body)(), 
 ...
)
{
  kaapi_thread_t* const thread = kaapi_self_thread();
  size_t total_size;
  kaapic_task_info_t* ti;
  va_list va_args;
  size_t wordsize;
  unsigned int k;
  uintptr_t* values;

  if (*nargs > KAAPIC_MAX_ARGS) return KAAPIF_ERR_EINVAL;

  /* fortran values are actually pointers and must be saved
     since they are no longer valid past this function.
     a slot is thus allocated for each eventual value in the
     thread stack and the type is converted into a TYPE_PTR.
  */

  total_size = offsetof(kaapic_task_info_t, args) +
    ((*nargs) * (sizeof(kaapic_arg_info_t) + sizeof(uintptr_t)));
  ti = kaapi_thread_pushdata_align(thread, total_size, sizeof(void*));

  /* last arg_info is actually the first value */
  values = (uintptr_t*)&ti->args[*nargs];

  ti->body  = body;
  ti->nargs = *nargs;

  va_start(va_args, body);
  for (k = 0; k < *nargs; ++k)
  {
    kaapic_arg_info_t* const ai = &ti->args[k];

    const int32_t mode  = *va_arg(va_args, int32_t*);
    void* addr		 = va_arg(va_args, void*);
    const int32_t count = *va_arg(va_args, int32_t*);
    int32_t type        = *va_arg(va_args, int32_t*);

    switch (mode)
    {
      case KAAPIC_MODE_R:  
        ai->mode = KAAPI_ACCESS_MODE_R; 
        break;
      case KAAPIC_MODE_W:  
        ai->mode = KAAPI_ACCESS_MODE_W; 
        break;
      case KAAPIC_MODE_RW: 
        ai->mode = KAAPI_ACCESS_MODE_RW; 
        break;
      case KAAPIC_MODE_V:  
        if (count >1) return KAAPIF_ERR_EINVAL;
	if (type != KAAPIC_TYPE_PTR)
	{
	  /* save into value space and change to TYPE_PTR */
	  values[k] = *(uintptr_t*)addr;
	  type = KAAPIC_TYPE_PTR;
	}
	else
	{
	  /* FORTRAN pointer is passed, but &ptr must be given to task */
	  values[k] = (uintptr_t)addr;
	}
	addr = (void*)(values + k);
	ai->mode = KAAPI_ACCESS_MODE_V;
        break;
      default: 
        return KAAPIF_ERR_EINVAL;
    }

    switch (type)
    {
      case KAAPIC_TYPE_CHAR:
        wordsize = sizeof(int);
        ai->format = kaapi_char_format;
        break ;

      case KAAPIC_TYPE_INT:
        wordsize = sizeof(int);
        ai->format = kaapi_int_format;
        break ;

      case KAAPIC_TYPE_REAL:
        wordsize = sizeof(float);
        ai->format = kaapi_float_format;
        break ;

      case KAAPIC_TYPE_DOUBLE:
        wordsize = sizeof(double);
        ai->format = kaapi_double_format;
        break ;

      case KAAPIC_TYPE_PTR:
        wordsize = sizeof(void*);
        ai->format = kaapi_voidp_format;
        break ;

      case KAAPIC_TYPE_ID:
        if (mode == KAAPIC_MODE_V) 
          return KAAPIF_ERR_EINVAL;

        /* should not have any format... fix the runtime to handle this case */
        wordsize = sizeof(void*);
        ai->format = kaapi_voidp_format;
        addr = *(void**)addr;
        break ;

      default:
        return KAAPIF_ERR_EINVAL;
    }
    
    kaapi_access_init( &ai->access, addr );
    if (mode == KAAPIC_MODE_V)
    {
      ai->access.version = (void*)(uintptr_t)addr;
    }

    ai->view = kaapi_memory_view_make1d(count, wordsize);
  }
  va_end(va_args);

  /* spawn the task */
  if (0== kaapic_spawn_ti( thread, 0, kaapic_dfg_body, ti ))
    return KAAPIF_SUCCESS;
  return KAAPIF_ERR_FAILURE;
}
