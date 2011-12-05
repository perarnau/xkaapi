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

#define CONFIG_USE_DELAY 0

#include "kaapi_impl.h"

#if CONFIG_USE_DELAY
#include <unistd.h>
#endif



/** Do rand selection 
*/
int kaapi_sched_select_victim_rand( 
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

      /* select a victim */
      if (kproc->fnc_selecarg[0] == 0) 
	kproc->fnc_selecarg[0] = (void*)(long)rand();

    redo_select:
      nbproc = kaapi_count_kprocessors;
      if (nbproc <=1) return EINVAL;
      victimid = rand_r( (unsigned int*)&kproc->fnc_selecarg ) % nbproc;

      /* Get the k-processor */    
      victim->kproc = kaapi_all_kprocessors[ victimid ];
      if (victim->kproc ==0) goto redo_select;

#if CONFIG_USE_DELAY
      /* wait, deduced from previous failed steal operations */
      /* kproc->fnc_selecarg[2] the delay to wait, if needed */
      const uintptr_t delayus = (uintptr_t)kproc->fnc_selecarg[2];
      if (delayus) usleep(delayus);
#endif /* CONFIG_USE_DELAY */

      break ;
    }

#if CONFIG_USE_DELAY
  case KAAPI_STEAL_SUCCESS:
    {
      /* reset failed steal count and delay us */
      kproc->fnc_selecarg[1] = 0;
      kproc->fnc_selecarg[2] = 0;
      break ;
    }
#endif /* CONFIG_USE_DELAY */

#if CONFIG_USE_DELAY
  case KAAPI_STEAL_FAILED:
    {
      const uintptr_t nsteals = (uintptr_t)kproc->fnc_selecarg[1];

      /* wait for failures to reach threshold */
      if (nsteals <= (kaapi_count_kprocessors * 8))
      {
	kproc->fnc_selecarg[1] = (void*)(nsteals + 1);
      }
      /* update delay */
      else
      {
	/* delayus set to 0 until failures reach kaapi_processor_count * 8
	   delayus initially set to delay_init
	   delayus incremented by delay_step per failure
	   delayus not incremented if greater than delay_thre
	   delayus reset on steal success
	 */

	static const uintptr_t delay_init = 1000;
	static const uintptr_t delay_step = 1000;
	static const uintptr_t delay_thre = 10000;

	uintptr_t delayus = (uintptr_t)kproc->fnc_selecarg[2];
        if (delayus == 0) delayus = delay_init;
	else if (delayus < delay_thre) delayus += delay_step;
	*((uintptr_t*)&kproc->fnc_selecarg[2]) = delayus;
      }

      break ;
    }
#endif /* CONFIG_USE_DELAY */

  default:
    {
      break ;
    }
  }

  return 0;
}
