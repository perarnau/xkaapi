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

typedef struct kaapi_aff_arg {
  kaapi_processor_id_t lastvictim;
  kaapi_processor_id_t nextvictim;
  uintptr_t            seed;
} kaapi_aff_arg;


/** Do rand selection: in case of failure, try to steal the previous last victim
*/
int kaapi_sched_select_victim_aff( 
    kaapi_processor_t* kproc, 
    kaapi_victim_t* victim, 
    kaapi_selecvictim_flag_t flag 
)
{
  kaapi_aff_arg* arg;

  kaapi_assert_debug( sizeof(kaapi_aff_arg) <= sizeof(kproc->fnc_selecarg) );

  arg = (kaapi_aff_arg*)&kproc->fnc_selecarg;

  if (flag == KAAPI_STEAL_FAILED)
  {
    arg->nextvictim = rand_r( (unsigned int*)&arg->seed );
    arg->lastvictim = arg->nextvictim;
    return 0;
  }

  if (flag == KAAPI_STEAL_SUCCESS)
  {
    /* success: try next to time initial depth */
    arg->nextvictim = arg->lastvictim;
    //arg->lastvictim = arg->nextvictim;
    return 0;
  }

  kaapi_assert_debug (flag == KAAPI_SELECT_VICTIM);

  if (arg->seed == 0) 
  {
    arg->seed = (uintptr_t)(long)rand();
    arg->lastvictim = rand_r( (unsigned int*)&arg->seed );
  }

redo_select:
  /* first: select in self set */
  //victim->kproc = kaapi_all_kprocessors[ arg->lastvictim % kaapi_count_kprocessors ];
  victim->kproc = kaapi_all_kprocessors[ 0 ];
  if (victim->kproc ==0) 
    goto redo_select;
  return 0;
}
