/*
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:21:00 2009
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


static int kaapi_select_victim_workload( 
  kaapi_processor_t* kproc __attribute__((unused)), 
  kaapi_victim_t* victim 
)
{  
  const int count = kaapi_count_kprocessors;
  kaapi_processor_t* const self_kproc = kaapi_get_current_processor();
  
  kaapi_processor_t* max_kproc = NULL;
  uint32_t max_workload = 1;
  int i;
  
  for (i = 0; i < count; ++i)
  {
    kaapi_processor_t* const cur_kproc = kaapi_all_kprocessors[i];
    
    if ((cur_kproc == NULL) || (cur_kproc == self_kproc))
      continue ;
    
    /* swap if more or equally loaded */
    {
      const uint32_t cur_workload = KAAPI_ATOMIC_READ(&cur_kproc->workload);
      
      if (cur_workload < max_workload)
        continue ;
      
      max_workload = cur_workload;
      max_kproc = cur_kproc;
    }
  }
  
  /* found a victim */
  if (max_kproc != NULL)
  {
    victim->kproc = max_kproc;
    return 0;
  }
  
  return EINVAL;
}


/** Do workload then rand selection 
 */
int kaapi_sched_select_victim_workload_rand( kaapi_processor_t* kproc, kaapi_victim_t* victim, kaapi_selecvictim_flag_t flag )
{
  int err;
  if (flag != KAAPI_SELECT_VICTIM) return 0;  
  do {
    err = kaapi_select_victim_workload( kproc, victim );
    if (err ==0) return 0;
        
  } while(1);
}


/* workload accessors
 */

void kaapi_set_workload(kaapi_processor_t* kproc, uint32_t workload)
{
  KAAPI_ATOMIC_WRITE(&kproc->workload, workload);
}


void kaapi_set_self_workload(uint32_t workload)
{
  kaapi_set_workload(kaapi_get_current_processor(), workload);
}


void kaapi_set_workload_by_kid(kaapi_processor_id_t kid, uint32_t workload)
{
  kaapi_set_workload(kaapi_all_kprocessors[kid], workload);
}
