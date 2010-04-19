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

/** Do one random selection at the given hierarchy level
*/
int kaapi_select_victim_rand_atlevel( kaapi_processor_t* kproc, int level, kaapi_victim_t* victim )
{  
  unsigned int victimid = 0;
  int nbproc;
  
  kaapi_assert_debug( kproc !=0 );
  kaapi_assert_debug( victim !=0 );

  if (kproc->hlevel < level) return EINVAL;

  victim->level = level;

  if (kproc->fnc_selecarg ==0) {
    kproc->fnc_selecarg = (void*)(long)rand();
/*TG: After very few experiments, it seems to better take random choice first
    victim->kproc = kaapi_all_kprocessors[ 0 ];
    return 0;
*/
  }

redo_select:
  nbproc = kproc->hlcount[level];
  if (nbproc <=1) return EINVAL;
#if 0
  victimid = rand_r( (unsigned int*)&kproc->fnc_selecarg ) % nbproc;
#else
/* \WARNING: test to bias the random generator */
  victimid = 0; //rand_r( (unsigned int*)&kproc->fnc_selecarg ) % (10*nbproc);
  if (victimid >= nbproc) victimid = 0;
#endif

  /* Get the k-processor */    
  victim->kproc = kaapi_all_kprocessors[ kproc->hkids[level][victimid] ];
  if (victim->kproc ==0) goto redo_select; 

  return 0;
}
