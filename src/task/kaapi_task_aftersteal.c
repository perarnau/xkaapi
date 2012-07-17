/*
** kaapi_task_signal.c
** xkaapi
** 
**
** Copyright 2009 INRIA.
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
#include <stdio.h>

/**
*/
void kaapi_aftersteal_body( void* taskarg, kaapi_thread_t* thread, kaapi_task_t*  task )
{
  unsigned int          i;
  size_t                count_params;
  const kaapi_format_t* fmt;   /* format of the task */
  const kaapi_format_t* fmt_param;
  kaapi_access_t        access_param;
  
  /* the task has been stolen: the body contains the original task body */
  kaapi_assert_debug( task->sp == taskarg );
  fmt = kaapi_format_resolvebybody( kaapi_task_getbody(task) );
  kaapi_assert_debug( fmt !=0 );

  count_params = kaapi_format_get_count_params(fmt, task->sp );

  /* for each access parameter, we have:
     - data that points to the original effective parameter
     - version that points to the data that should have been used during the execution of the stolen task
     If the access is a W (or CW) access then, version contains the newly produced data.
     The purpose here is to merge the version into the original data
        - if W mode -> copy + free of the data
        - if CW mode -> accumulation (data is the left side of the accumulation, version the right side)
  */
  for (i=0; i<count_params; ++i)
  {
    kaapi_access_mode_t mode = kaapi_format_get_mode_param(fmt, i, taskarg);
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE( mode ); /* forget extra flags */
    if (m == KAAPI_ACCESS_MODE_V)
      continue;

    if (KAAPI_ACCESS_IS_ONLYWRITE(m) || KAAPI_ACCESS_IS_CUMULWRITE(m) )
    {
      fmt_param    = kaapi_format_get_fmt_param(fmt, i, taskarg);
      access_param = kaapi_format_get_access_param(fmt, i, taskarg);

      /* if m == W || CW and data == version it means that data used by W was not copied due 
         to non WAR dependency or no CW access 
      */
      if (access_param.data != access_param.version )
      {
        /* add an assign + dstor function will avoid 2 calls to function, especially for basic types which do not
           required to be dstor.
        */
        if (!KAAPI_ACCESS_IS_CUMULWRITE(m))
        {
          kaapi_memory_view_t view_dest = kaapi_format_get_view_param(fmt, i, taskarg);
          kaapi_memory_view_t view_src;
          
          /* in case of WAR resolution, view_src is == to reallocation of view_src */
          view_src = view_dest;
          kaapi_memory_view_reallocated( &view_src );

          /* here we assume that if no assignment, then this is a pod type, so we use memcpy version for cpu2cpu
             If version is produced remotely... version should store the location...
          */
          if (fmt_param->assign ==0)
          {
            kaapi_pointer_t     ptr_dest;
            kaapi_void2pointer(&ptr_dest, access_param.data);
            kaapi_memory_write_cpu2cpu(ptr_dest, &view_dest, access_param.version, &view_src);
          }
          else
            (*fmt_param->assign)( access_param.data, &view_dest, access_param.version, &view_src );
        }
        else {
          /* this is a CW mode : do reduction except if it is INPLACE */
          if ((mode & KAAPI_ACCESS_MODE_IP) ==0)
            kaapi_format_reduce_param( fmt, i, taskarg, access_param.version );
        }
        if ((mode & KAAPI_ACCESS_MODE_IP) ==0)
        {
          if (fmt_param->dstor !=0) (*fmt_param->dstor) ( access_param.version );
          free(access_param.version);
        }
      }
      access_param.version = 0;
#if defined(KAAPI_DEBUG)
      kaapi_format_set_access_param(fmt, i, taskarg, &access_param);
#endif      
    }
  }
}
