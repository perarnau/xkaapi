/*
 ** xkaapi
 ** 
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributor :
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
#if defined(KAAPI_DEBUG_LOURD)
#include <unistd.h>
#endif

/* Compute if the task with arguments pointed by sp and with format task_fmt is ready
 Return the number of non ready data
 */
size_t kaapi_task_computeready( 
  kaapi_task_t*         task __attribute__((unused)),
  void*                 sp, 
  const kaapi_format_t* task_fmt, 
  unsigned int*         war_param, 
  unsigned int*         cw_param, 
  kaapi_hashmap_t*      map 
)
{
  size_t count_params;
  size_t wc;
  unsigned int i;
  
  count_params = wc = kaapi_format_get_count_params(task_fmt, sp); 
  
  for (i=0; i<count_params; ++i)
  {
    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(  kaapi_format_get_mode_param(task_fmt, i, sp) );
    if (m == KAAPI_ACCESS_MODE_V)
    {
      --wc;
      continue;
    }
    kaapi_access_t access = kaapi_format_get_access_param(task_fmt, i, sp);
    
    /* */
    kaapi_gd_t* gd = &kaapi_hashmap_findinsert( map, access.data )->u.value;
    
    /* compute readyness of access
       - note that stack access, even if it as R|W flag is always concurrent
    */
    if (   KAAPI_ACCESS_IS_ONLYWRITE(m)
        || (gd->last_mode == KAAPI_ACCESS_MODE_VOID)
        || (KAAPI_ACCESS_IS_STACK(m))
        || (KAAPI_ACCESS_IS_CONCURRENT(m, gd->last_mode))
       )
    {
      --wc;
      if (  (KAAPI_ACCESS_IS_ONLYWRITE(m) && KAAPI_ACCESS_IS_READ(gd->last_mode))
         || (KAAPI_ACCESS_IS_CUMULWRITE(m) && KAAPI_ACCESS_IS_CONCURRENT(m,gd->last_mode)) 
         || (KAAPI_ACCESS_IS_STACK(m))
      )
      {
        /* stack data are reused through recursive task execution and copied on steal but never merged */
        *war_param |= 1<<i;
      }
      if (KAAPI_ACCESS_IS_CUMULWRITE(m))
        *cw_param |= 1<<i;
    }
    
    /* update map information for next access if no set */
    if (gd->last_mode == KAAPI_ACCESS_MODE_VOID)
      gd->last_mode = m;
    
    /* Datum produced by aftersteal_task may be made visible to thief in order to augment
       the parallelism by breaking chain of versions (W->R -> W->R ), the second W->R may
       be used (the middle R->W is splitted -renaming is also used in other context-).
       But we do not take into account of this extra parallelism.
     */
  }
  return wc;
}
