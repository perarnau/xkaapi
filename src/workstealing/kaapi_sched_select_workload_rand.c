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


/** Do rand selection 
    adapted from kaapi_sched_select_victim_rand.c
    steal a workloaded processor at random BUT try
    zero first.
*/
int kaapi_sched_select_victim_workload_rand
(
  kaapi_processor_t* kproc, 
  kaapi_victim_t* victim, 
  kaapi_selecvictim_flag_t flag 
)
{
  switch (flag)
  {
  case KAAPI_SELECT_VICTIM:
    {
      int nbproc, victimid;
      unsigned int ntries = 0;
      unsigned int do_zero = 1;

      /* select a victim */
      if (kproc->fnc_selecarg[0] == 0) 
	kproc->fnc_selecarg[0] = (uintptr_t)(long)rand();

    redo_select:
      nbproc = kaapi_count_kprocessors;
      if (nbproc <=1) return EINVAL;

      if (do_zero == 1)
      {
	victimid = 0;
	do_zero = 0;
      }
      else
      {
	victimid = rand_r( (unsigned int*)&kproc->fnc_selecarg ) % nbproc;
      }

      /* Get the k-processor */    
      victim->kproc = kaapi_all_kprocessors[ victimid ];
      if (victim->kproc ==0) goto redo_select;

      /* first workload barrier */
      if (ntries < (kaapi_count_kprocessors * 2))
      {
	++ntries;

	if (KAAPI_ATOMIC_READ(&victim->kproc->workload) == 0)
	  goto redo_select;
      }

      break ;
    }

  default:
    {
      break ;
    }
  }

  return 0;
}


/* workload accessors
 */

void kaapi_set_workload(kaapi_processor_t* kproc, unsigned long workload)
{
  KAAPI_ATOMIC_WRITE(&kproc->workload, workload);
}


void kaapi_set_self_workload(unsigned long workload)
{
  kaapi_set_workload(kaapi_get_current_processor(), workload);
}
