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

typedef struct kaapi_hier_arg {
  short         depth;
  short         depth_min;
  unsigned int  index;
  unsigned int  seed;
} kaapi_hier_arg;


/** Do rand selection 
*/
int kaapi_sched_select_victim_hierarchy( kaapi_processor_t* kproc, kaapi_victim_t* victim, kaapi_selecvictim_flag_t flag )
{
  int victimid;
  kaapi_hier_arg* arg;
  kaapi_onelevel_t* level;

  kaapi_assert_debug( sizeof(kaapi_hier_arg) <= sizeof(kproc->fnc_selecarg) );

  if (kproc->hlevel.depth ==0) 
    return kaapi_sched_select_victim_rand(kproc, victim, flag );

  kaapi_assert_debug( sizeof(kaapi_hier_arg)  <= sizeof(kproc->fnc_selecarg) );
  arg = (kaapi_hier_arg*)&kproc->fnc_selecarg;

  if (flag == KAAPI_STEAL_FAILED)
  {
    level = &kproc->hlevel.levels[arg->depth];
    if (++arg->index >= level->nkids)
    {
      ++arg->depth;
      arg->index = 0;
      if (arg->depth == kproc->hlevel.depth) 
        arg->depth = arg->depth_min-1;
    }
    return 0;
  }
  if (flag == KAAPI_STEAL_SUCCESS)
  {
    arg->depth = arg->depth_min-1;
    arg->index = 0;
  }

  if (arg->depth_min ==-1) 
    return kaapi_sched_select_victim_rand(kproc, victim, flag );

  if (flag != KAAPI_SELECT_VICTIM) 
    return 0;

  if (arg->depth_min ==0) 
  {
    unsigned int i;
    arg->seed = rand();
    level = &kproc->hlevel.levels[arg->depth_min];
    /* set to the min level where at least 2 threads */
    while (level->nkids ==1)
    {
      ++arg->depth_min;
      if (arg->depth_min >= kproc->hlevel.depth)
      {
        arg->depth_min = -1;
        break;
      }
      level = &kproc->hlevel.levels[arg->depth_min];
    }
    printf("kid:%i, cpu:%i mindepth:%i ", kproc->kid, kproc->cpuid, arg->depth_min);
    for (i=0; i<level->nkids; ++i)
      printf(", kid: %i ", level->kids[i] ); 
    printf("\n");
    fflush(stdout);
    /* always used with -1 */
    ++arg->depth_min;
  }

redo_select:
  level = &kproc->hlevel.levels[arg->depth];
  victimid = level->kids[ rand_r(&arg->seed) % level->nkids];  
  victim->kproc = kaapi_all_kprocessors[ victimid ];
  if (victim->kproc ==0) 
    goto redo_select;
  return 0;
}
