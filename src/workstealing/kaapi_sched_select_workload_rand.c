/*
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

#define MAX_SKPROC 4

static int kaapi_select_victim_workload( 
  kaapi_processor_t* self_kproc,
  kaapi_victim_t* victim 
)
{  
  int32_t cur_workload;
  int k;
  
  while (1)
  {
    kaapi_processor_t* max_kproc =0;
    uint32_t max_workload= 0;
    
    for (k = 0; k < 3; ++k)
    {
      kaapi_processor_t* const victim_kproc = kaapi_all_kprocessors[ 
          rand_r( (unsigned int*)&self_kproc->seed ) %kaapi_count_kprocessors ];
      
      if ((victim_kproc == NULL) || (victim_kproc == self_kproc))
        continue ;
      
      /* swap if more or equally loaded */
      cur_workload = kaapi_processor_get_workload(victim_kproc);
      
      if (cur_workload > max_workload)
      {
        max_kproc    = victim_kproc;
        max_workload = cur_workload;
      }
    }
    
    victim->kproc = max_kproc;
    if  (max_kproc != 0) return 0;
  }
  return EINVAL;
}


/** Do workload then rand selection 
 */
int kaapi_sched_select_victim_workload_rand( 
  kaapi_processor_t* kproc, 
  kaapi_victim_t* victim, 
  kaapi_selecvictim_flag_t flag 
)
{
  int err;
  if (flag != KAAPI_SELECT_VICTIM) return 0;  
  err = kaapi_select_victim_workload( kproc, victim );
  return err;
}


/* workload accessors
 */

void kaapi_set_workload(kaapi_processor_t* kproc, uint32_t workload)
{
  KAAPI_ATOMIC_WRITE(&kproc->workload, workload);
}


void kaapi_set_self_workload(uint32_t workload)
{
  kaapi_processor_set_workload(kaapi_get_current_processor(), workload);
}


#if 0
void kaapi_set_workload_by_kid(kaapi_processor_id_t kid, uint32_t workload)
{
  kaapi_set_workload(kaapi_all_kprocessors[kid], workload);
}
#endif