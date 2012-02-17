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
  short         nfailed;
  short         depth;     /* current depth where to select */
  short         depth_min;
  unsigned int  index;     /* 0: self set, 1: notself set */
  unsigned int  seed;
} kaapi_hier_arg;


/** Do rand selection 
*/
int kaapi_sched_select_victim_hwsn( 
    kaapi_processor_t* kproc, 
    kaapi_victim_t* victim, 
    kaapi_selecvictim_flag_t flag 
)
{
  int victimid;
  kaapi_hier_arg* arg;
  kaapi_onelevel_t* level;

  kaapi_assert_debug( sizeof(kaapi_hier_arg) <= sizeof(kproc->fnc_selecarg) );

  arg = (kaapi_hier_arg*)&kproc->fnc_selecarg;

  if ((kproc->hlevel.depth ==0) || (arg->depth_min ==-1))
  { /* no hierarchy: like random flat selection */
    return kaapi_sched_select_victim_rand(kproc, victim, flag );
  }

  if (flag == KAAPI_STEAL_FAILED)
  {
    int nset;
    level = &kproc->hlevel.levels[arg->depth];
    nset = (arg->index ==0 ? level->nsize : level->nnotself);

    ++arg->nfailed;

    /* failed: try on not self set before go up */
    if  (arg->nfailed >= 1+nset/4)
    {
      arg->nfailed = 0;
      if (arg->index ==0)
        arg->index   = 1;
      else {
        ++arg->depth;
        arg->index   = 0;
      }
      if (arg->depth == kproc->hlevel.depth) 
        arg->depth = kproc->hlevel.depth;
    }
    return 0;
  }

  if (flag == KAAPI_STEAL_SUCCESS)
  {
    /* success: try next to time initial depth */
    arg->depth   = 0;
    arg->nfailed = 0;
    arg->index   = 0;
    return 0;
  }

  kaapi_assert_debug (flag == KAAPI_SELECT_VICTIM);

redo_select:
  /* first: select in self set */
  level = &kproc->hlevel.levels[arg->depth];
  if (arg->index ==0)
    victimid = level->kids[ rand_r(&arg->seed) % level->nkids];
  else
    victimid = level->notself[ rand_r(&arg->seed) % level->nnotself];

  victim->kproc = kaapi_all_kprocessors[ victimid ];
  if (victim->kproc ==0) 
    goto redo_select;
  return 0;
}
