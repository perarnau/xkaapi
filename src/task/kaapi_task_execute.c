/*
** kaapi_task_standard.c
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
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
#include "kaapi_impl.h"
#include <unistd.h>
#include <sys/mman.h>

#if 1// !defined (KAAPI_DEBUG)

/* the method should be redesigned in order to keep pre-processing before execution and post-processing after sub-tasks 
finish their executions
*/
void _kaapi_task_execute_with_control( kaapi_task_body_internal_t body, void* sp, kaapi_frame_t* fp, kaapi_task_t* pc )
{
  body(sp, fp, pc);
}
#else
/* this is the debug version: 
   - allocate temporaries for read access modes with only PROT_READ access
   - allocate temporaries for write access modes with only PROT_WRITE access
   - allocate temporaries for cumulative write access modes with only PROT_WRITE|PROT_READ accesses
   - allocate temporaries for read-write access modes with only PROT_WRITE|PROT_READ accesses
   - Stack variables are managed as read-write

   All temporary buffers are allocated aligned to pagesize with extra guard pages to ensure no out-of-range accesses.
*/
typedef struct kaapi_control_access_t {
  kaapi_access_t      src;
  kaapi_memory_view_t src_view;
  kaapi_access_t      copy;
  kaapi_memory_view_t copy_view;
  kaapi_access_mode_t mode;
} kaapi_control_access_t;

void _kaapi_task_execute_with_control( 
  kaapi_task_body_internal_t body, void* sp, kaapi_frame_t* fp, kaapi_task_t* pc 
)
{
  const kaapi_format_t* fmt = kaapi_format_resolvebybody((kaapi_task_bodyid_t)body);
  int i, nargs;
  kaapi_control_access_t* access;

  if (fmt == 0)
  {
    printf("[kaapi_task_execute] task without format\n");
    body(sp,fp,pc);
    return;
  }

  size_t psz    = getpagesize();
  nargs = kaapi_format_get_count_params(fmt, sp);
  access = (kaapi_control_access_t*)alloca( sizeof(kaapi_control_access_t)*nargs );
  for (i=0; i<nargs; ++i)
  {
    access[i].mode = kaapi_format_get_mode_param(fmt, i, sp);
    if (access[i].mode == KAAPI_ACCESS_MODE_V)
      continue;

    access[i].src = kaapi_format_get_access_param(fmt, i, sp);
    access[i].src_view = kaapi_format_get_view_param(fmt, i, sp);

    access[i].copy_view = access[i].src_view;
    kaapi_memory_view_reallocated(&access[i].copy_view);
    /* allocation ensures pages size multiple */
    access[i].copy.data = kaapi_alloc_protect( kaapi_memory_view_size(&access[i].copy_view) );

    kaapi_access_mode_t mode = KAAPI_ACCESS_GET_MODE(access[i].mode);
    switch (mode) 
    {
      case KAAPI_ACCESS_MODE_S:
      case KAAPI_ACCESS_MODE_RW:
      case KAAPI_ACCESS_MODE_R:
      case KAAPI_ACCESS_MODE_CW:
      { /* recopy data */
        kaapi_memory_write_cpu2cpu( kaapi_make_localpointer(access[i].copy.data), &access[i].copy_view, 
                                    kaapi_make_localpointer(access[i].src.data), &access[i].src_view );
      } break;

      case KAAPI_ACCESS_MODE_W:
      { /* do nothing: write ! */
      } break;

      case KAAPI_ACCESS_MODE_T:
      { /* do nothing: temporary data */
      } break;
      
      default:
      break;
    }

    /* put protection mode */
    size_t npages = (kaapi_memory_view_size(&access[i].copy_view) + psz -1)/psz;

    switch (mode) 
    {
      case KAAPI_ACCESS_MODE_T:
      case KAAPI_ACCESS_MODE_S:
      case KAAPI_ACCESS_MODE_RW:
      case KAAPI_ACCESS_MODE_CW:
      { /* read and write access here */
        mprotect( access[i].copy.data, npages, PROT_WRITE|PROT_READ );
      } break;

      case KAAPI_ACCESS_MODE_R:
      { /* read access here */
        mprotect( access[i].copy.data, npages, PROT_READ );
      } break;

      case KAAPI_ACCESS_MODE_W:
      { /* read access here */
        mprotect( access[i].copy.data, npages, PROT_WRITE );
      } break;

      default:
      break;
    }
    
    /* replace parameter */
    kaapi_format_set_access_param(fmt, i, sp, &access[i].copy);
    kaapi_format_set_view_param(fmt, i, sp, &access[i].copy_view);
  }
  
  /* Call the function with new parameter */
  body( sp, fp, pc );

  for (i=0; i<nargs; ++i)
  {
    kaapi_access_mode_t mode = KAAPI_ACCESS_GET_MODE(access[i].mode);
    if (mode == KAAPI_ACCESS_MODE_V)
      continue;

    size_t npages = (kaapi_memory_view_size(&access[i].copy_view) + psz -1)/psz;
    switch (mode) 
    {
      case KAAPI_ACCESS_MODE_W:
        mprotect( access[i].copy.data, npages, PROT_READ );
      case KAAPI_ACCESS_MODE_S:
      case KAAPI_ACCESS_MODE_RW:
      case KAAPI_ACCESS_MODE_CW:
      { /* recopy data to original location */
        kaapi_memory_write_cpu2cpu( kaapi_make_localpointer(access[i].src.data), &access[i].src_view,
                                    kaapi_make_localpointer(access[i].copy.data), &access[i].copy_view );
      } break;

      case KAAPI_ACCESS_MODE_T:
      { /* do nothing: temporary data */
      } break;
      
      default:
      break;
    }
    
    /* restore parameter */
    kaapi_format_set_access_param(fmt, i, sp, &access[i].src);
    kaapi_format_set_view_param(fmt, i, sp, &access[i].src_view);

    /* free temporary */
    kaapi_free_protect( access[i].copy.data );
  }
}
#endif
