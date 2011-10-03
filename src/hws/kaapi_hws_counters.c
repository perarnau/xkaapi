/*
** kaapi_hws_counter.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
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

#include <string.h>
#include "kaapi_impl.h"
#include "kaapi_hws.h"
#include "kaapi_ws_queue.h"


#if CONFIG_HWS_COUNTERS


static inline unsigned int is_kid_used(kaapi_processor_id_t kid)
{
  return (kaapi_default_param.kid2cpu[kid] != -1U);
}

static inline unsigned int kid_to_cpu(kaapi_processor_id_t kid)
{
  return kaapi_default_param.kid2cpu[kid];
}

void kaapi_hws_print_counters(void)
{
  /* walk the tree and print for all blocks */
  kaapi_hws_level_t* level;
  kaapi_ws_block_t* block;
  kaapi_processor_id_t kid;
  kaapi_hws_levelid_t levelid;
  unsigned int i;
  unsigned long val;

  printf("= HWS_STEAL_COUNTERS\n");

  for (levelid = 0; levelid < (int)hws_level_count; ++levelid)
  {
    level = &hws_levels[levelid];

    /* always display flat level: see steal_block_leaves */
    if (levelid != KAAPI_HWS_LEVELID_FLAT)
      if (!kaapi_hws_is_levelid_set(levelid)) continue ;

    printf("\n- %s\n", kaapi_hws_levelid_to_str(levelid));

    /* print used kids */
    printf("CPU ");
    i = 3;
    for (kid = 0; kid < KAAPI_MAX_PROCESSOR; ++kid)
      if (is_kid_used(kid))
      {
	printf("%08u ", kid_to_cpu(kid));
	i += 9;
      }
    printf("\n");
    for (; i; --i) printf("-");
    printf("\n");

    /* print steal counters */
    for (i = 0; i < level->block_count; ++i)
    {
      block = level->blocks + i;

      /* block count */
      printf("#%02u", i);

      /* dont print steal counters if FLAT not set */
      if (levelid == KAAPI_HWS_LEVELID_FLAT)
	if (!kaapi_hws_is_levelid_set(KAAPI_HWS_LEVELID_FLAT))
	{
	  printf("\n");
	  continue ;
	}

      for (kid = 0; kid < KAAPI_MAX_PROCESSOR; ++kid)
      {
	if (is_kid_used(kid) == 0) continue ;

	val = KAAPI_ATOMIC_READ(&block->queue->steal_counters[kid]);
	if (val) printf(" %08lu", val);
	else printf("         ");
      }

      printf("\n");
    }
    printf("\n");
  }

  /* flat level */
  printf("= HWS_POP_COUNTERS\n");

  level = &hws_levels[KAAPI_HWS_LEVELID_FLAT];

  /* print used kids */
  printf("CPU ");
  i = 3;
  for (kid = 0; kid < KAAPI_MAX_PROCESSOR; ++kid)
    if (is_kid_used(kid))
    {
      printf("%08u ", kid_to_cpu(kid));
      i += 9;
    }
  printf("\n");
  for (; i; --i) printf("-");
  printf("\n");

  printf("    ");
  for (kid = 0; kid < KAAPI_MAX_PROCESSOR; ++kid)
  {
    if (is_kid_used(kid) == 0) continue ;

    block = level->kid_to_block[kid];

    /* print pop counters */
    val = KAAPI_ATOMIC_READ(&block->queue->pop_counter);
    if (val) printf("%08lu ", val);
    else printf("         ");
  }
  printf("\n");
}


#endif /* CONFIG_HWS_COUNTERS */
