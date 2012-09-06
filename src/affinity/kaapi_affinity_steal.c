/*
 ** xkaapi
 ** 
 **
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 **
 ** joao.lima@imag.fr
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
#include "kaapi_tasklist.h"


//#undef KAAPI_DEBUG_INST
//#define KAAPI_DEBUG_INST(x) x

/* return the first td that can be executed on arch 
*/
kaapi_taskdescr_t* kaapi_steal_by_affinity_first( const kaapi_processor_t* thief, kaapi_taskdescr_t* td )
{
  int arch = kaapi_processor_get_type(thief);
  
  /* only steal for the righ processor arch or if fmt ==0 (means internal task) */
  while ((td != 0) 
      && ( kaapi_task_getbody(td->task) != (kaapi_task_body_t)kaapi_staticschedtask_body )
      && (   !kaapi_task_has_arch(td->task,arch) 
          || ((td->fmt !=0) && (kaapi_format_get_task_body_by_arch(td->fmt, arch) ==0)) ) )
  {
    td = td->prev;
  }
  return td;
}


/*
*/
kaapi_taskdescr_t* kaapi_steal_by_affinity_maxctpath( const kaapi_processor_t* thief, kaapi_taskdescr_t* td )
{
  kaapi_taskdescr_t* td_max_date = kaapi_steal_by_affinity_first(thief, td);
  if (td_max_date ==0) 
    return 0;

  int arch = kaapi_processor_get_type(thief);

  /* only steal for the righ processor arch or if fmt ==0 (means internal task) */
  while (td != 0)
  {
    if ((td->fmt ==0) || (kaapi_task_has_arch(td->task,arch) && (kaapi_format_get_task_body_by_arch(td->fmt, arch) !=0)))
    {
      if (td->u.acl.date > td_max_date->u.acl.date)
        td_max_date = td;
    }
    td = td->prev;
  }
  return td_max_date;
}


/*
*/
kaapi_taskdescr_t* kaapi_steal_by_affinity_maxhit( const kaapi_processor_t* thief, kaapi_taskdescr_t* td )
{
  uint64_t hit, hitmax = 0;
  kaapi_taskdescr_t* tdhitmax = 0;

KAAPI_DEBUG_INST(
  kaapi_taskdescr_t* tdfirst_forarch = 0;
  uint64_t hitfirst = 0;
  int cnt_task =  0;
)

  int arch = kaapi_processor_get_type(thief);

  /* only steal for the righ processor arch or if fmt ==0 (means internal task) */
  while (td != 0)
  {
KAAPI_DEBUG_INST(
    ++cnt_task;
)
    if (td->fmt ==0) /* internal task: valid on any arch */
    {
      if (tdhitmax ==0)
      {
        hitmax   = 0;
        tdhitmax = td;
      }
KAAPI_DEBUG_INST(
      if (tdfirst_forarch ==0) 
        tdfirst_forarch = td;
)
    }
    else if (kaapi_task_has_arch(td->task,arch) && (kaapi_format_get_task_body_by_arch(td->fmt, arch) !=0))
    {
      hit = kaapi_data_get_affinity_hit_size( thief, td );
      if (tdhitmax ==0)
      {
        hitmax   = hit;
        tdhitmax = td;
      }
      else if (hit > hitmax)
      {
        hitmax   = hit;
        tdhitmax = td;
      }

KAAPI_DEBUG_INST(
      if (tdfirst_forarch ==0) 
      {
        hitfirst = hit;
        tdfirst_forarch = td;
      }
)
    }
    td = td->prev;
  }

KAAPI_DEBUG_INST(
  if ((tdhitmax !=0) && (hitmax > hitfirst))
    printf("Arch: %s  Hit Max: %lu, default hit:%lu\n", (arch == KAAPI_PROC_TYPE_CPU ? "CPU" : "GPU"), 
        (unsigned long)hitmax, (unsigned long)hitfirst
    );
)

  return tdhitmax;
}


kaapi_taskdescr_t* kaapi_steal_by_affinity_writer( const kaapi_processor_t* thief, kaapi_taskdescr_t* td )
{
  kaapi_taskdescr_t* td_curr= kaapi_steal_by_affinity_first(thief, td);
  if (td_curr==0) 
    return 0;

  int arch = kaapi_processor_get_type(thief);

  /* only steal for the righ processor arch or if fmt ==0 (means internal task) */
  while (td != 0)
  {
    if ((td->fmt != 0) && kaapi_task_has_arch(td->task,arch) && (kaapi_format_get_task_body_by_arch(td->fmt, arch) !=0))
    {
      if(kaapi_data_get_affinity_is_valid_writer(thief,td)) {
	return td;
      }
    }
    td = td->prev;
  }
  return td_curr;
}

