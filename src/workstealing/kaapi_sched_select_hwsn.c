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

typedef struct kaapi_hier_arg {
  short         init;     /* 0 iff not init  */
  short         policy;  /* 1: local, 2: rand */
  int           nfailed;
  unsigned int  seed;
} kaapi_hier_arg;


/** 
*/
int kaapi_sched_select_victim_hwsn( 
    kaapi_processor_t* kproc, 
    kaapi_victim_t* victim, 
    kaapi_selecvictim_flag_t flag 
)
{
  int victimid;
  int nbproc;
  int rr;
  kaapi_hier_arg* arg;
  kaapi_onelevel_t* level;

  kaapi_assert_debug( sizeof(kaapi_hier_arg) <= sizeof(kproc->fnc_selecarg) );

  arg = (kaapi_hier_arg*)&kproc->fnc_selecarg;

  if ((kproc->hlevel.depth ==0) || (arg->init ==-1))
  { /* no hierarchy: like random flat selection */
    return kaapi_sched_select_victim_rand(kproc, victim, flag );
  }


  if (flag == KAAPI_STEAL_FAILED)
  {
    ++arg->nfailed;
    if (arg->nfailed > 3)
      arg->policy = 2; /* random */
    return 0;
  }

  if (flag == KAAPI_STEAL_SUCCESS)
  {
    if (arg->policy !=1) arg->policy =1; 

    /* success: try next to time initial depth */
    arg->nfailed  = 0;
    return 0;
  }

  kaapi_assert_debug (flag == KAAPI_SELECT_VICTIM);
  if (arg->init ==0)
  {
    arg->init     = 1;
    arg->policy   = 2;
    arg->nfailed  = 0;
    arg->seed     = rand();
  }

  nbproc = kaapi_count_kprocessors;
  if (nbproc <=1) 
    return EINVAL;

redo_select:
  rr = rand_r(&arg->seed);
  /* first: select in self set */
  if (arg->policy ==1)
  {
    level = &kproc->hlevel.levels[0];
    victimid = level->notself[ rr % level->nnotself];
  }
  else
    victimid = rr % nbproc;

  victim->kproc = kaapi_all_kprocessors[ victimid ];
  if (victim->kproc ==0) 
    goto redo_select;

  return 0;
}
