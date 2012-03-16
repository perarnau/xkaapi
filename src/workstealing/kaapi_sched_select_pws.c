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
#include <stdlib.h>

/* This is an very experimental code. Do not is your are not developping
   the code.
   Principle in this version:
   - the cores are splited into two sets
      - the set of local cores LCS
      - the set of remote cores RCS
   - random selection with biased random generator in order
   to have about p chances to select cores in LCS and (1-p) in RCS.
   Algorithm:
      - r = rand() % (#RCS + #LCS*(1+P))
      - if (r < #RCS) -> select victim kaapi_all_kprocessors[r]
      - select in #LCS -> select victim ->self[ (r-#RCS) / #LCS]
      
*/

typedef struct kaapi_hier_arg {
  int            level;
  unsigned int   seed;
} kaapi_hier_arg;


/** PWS:
*/
int kaapi_sched_select_victim_pws( 
      kaapi_processor_t* kproc, 
      kaapi_victim_t* victim, 
      kaapi_selecvictim_flag_t flag 
)
{
  int victimid, count, r;
  kaapi_hier_arg* arg;
  kaapi_onelevel_t* level;

  kaapi_assert_debug( sizeof(kaapi_hier_arg) <= sizeof(kproc->fnc_selecarg) );

  if (kproc->hlevel.depth ==0) 
  { /* no hierarchy: do random flat selection */
    return kaapi_sched_select_victim_rand(kproc, victim, flag );
  }

  kaapi_assert_debug( sizeof(kaapi_hier_arg)  <= sizeof(kproc->fnc_selecarg) );
  arg = (kaapi_hier_arg*)&kproc->fnc_selecarg;

  if (flag == KAAPI_STEAL_FAILED)
  {
    if (arg->level <2)
      ++arg->level;
    return 0;
  }

  if (flag == KAAPI_STEAL_SUCCESS)
  {
    arg->level = 0;
    return 0;
  }
  
  if (arg->seed ==0)
  {
    arg->level = 2; /* for idfreeze test */
    arg->seed = rand();
  }

  kaapi_assert_debug (flag == KAAPI_SELECT_VICTIM);

redo_select:
  level = &kproc->hlevel.levels[arg->level];
  count = level->nkids + level->nnotself;
  r = rand_r(&arg->seed) % count;
  if (r < 0.8*count) {
    /* kids set */
    victimid  = level->kids[ r % level->nkids ];
  }
  else {
    /* notself set */
    victimid  = level->notself[ r % level->nnotself ];
  }
  victim->kproc = kaapi_all_kprocessors[ victimid ];
  if (victim->kproc ==0) 
    goto redo_select;
  return 0;
}
