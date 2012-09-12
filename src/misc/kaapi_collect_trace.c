/*
** kaapi_init.c
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com 
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
#include <inttypes.h> 

#if defined(KAAPI_USE_PERFCOUNTER)
#include <signal.h>
#endif


void kaapi_collect_trace(void)
{
#if defined(KAAPI_USE_PERFCOUNTER)
  int i;
  uint64_t cnt_tasks;
  uint64_t cnt_stealreqok;
  uint64_t cnt_stealreq;
  uint64_t cnt_stealop;
  uint64_t cnt_stealin;
  uint64_t cnt_suspend;
  uint64_t cnt_comm_out;
  uint64_t cnt_comm_in;
  double t_sched;
  double t_preempt;
  double t_1;
  double t_tasklist;
  
#if defined(KAAPI_USE_PAPI)
  uint64_t papicnt[2][KAAPI_PERF_ID_PAPI_MAX];
  memset( papicnt, 0, sizeof(papicnt) );
#endif

  cnt_tasks       = 0;
  cnt_stealreqok  = 0;
  cnt_stealreq    = 0;
  cnt_stealop     = 0;
  cnt_stealin     = 0;
  cnt_suspend     = 0;
  cnt_comm_in	  = 0;
  cnt_comm_out	  = 0;

  t_sched         = 0;
  t_preempt       = 0;
  t_1             = 0;
  t_tasklist      = 0;

  for (i=0; i<kaapi_count_kprocessors; ++i)
  {
    kaapi_event_closebuffer( kaapi_all_kprocessors[i]->eventbuffer );
    kaapi_all_kprocessors[i]->eventbuffer = 0;
    
    cnt_tasks +=      KAAPI_PERF_REG_READALL(kaapi_all_kprocessors[i], KAAPI_PERF_ID_TASKS);
    cnt_stealreqok += KAAPI_PERF_REG_READALL(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALREQOK);
    cnt_stealreq +=   KAAPI_PERF_REG_READALL(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALREQ);
    cnt_stealop +=    KAAPI_PERF_REG_READALL(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALOP);
    cnt_stealin +=    KAAPI_PERF_REG_READALL(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALIN);
    cnt_suspend +=    KAAPI_PERF_REG_READALL(kaapi_all_kprocessors[i], KAAPI_PERF_ID_SUSPEND);
    t_sched +=        1e-9*(double)KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_T1);
    t_preempt +=      1e-9*(double)KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_TPREEMPT);
    t_1 +=            1e-9*(double)KAAPI_PERF_REG_USR(kaapi_all_kprocessors[i], KAAPI_PERF_ID_T1); 
    t_tasklist +=     1e-9*(double)KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_TASKLISTCALC);

    cnt_comm_out +=   KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_COMM_OUT);
    cnt_comm_in +=   KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_COMM_IN);

#if defined(KAAPI_USE_PAPI)
    for (int cnt=KAAPI_PERF_ID_PAPI_BASE; cnt<kaapi_mt_perf_counter_num(); ++cnt)
    {
      papicnt[0][cnt-KAAPI_PERF_ID_PAPI_BASE] += KAAPI_PERF_REG_USR(kaapi_all_kprocessors[i], cnt);
      papicnt[1][cnt-KAAPI_PERF_ID_PAPI_BASE] += KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], cnt);
    }
#endif    

    /* */
    if (kaapi_default_param.display_perfcounter) //( && (kaapi_count_kprocessors <4))
    {

      printf("----- Performance counters, core    : %i\n", i);
      printf("Total number of tasks executed      : %"PRIi64 ", %" PRIi64 "\n",
        KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_TASKS),
        KAAPI_PERF_REG_USR(kaapi_all_kprocessors[i], KAAPI_PERF_ID_TASKS)
      );
      printf("Total number of steal OK requests   : %"PRIi64 ", %" PRIi64 "\n",
        KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALREQOK),
        KAAPI_PERF_REG_USR(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALREQOK)
      );
#if 0
      printf("Total number of steal OK requests   : %"PRIi64"\n",
        KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALREQOK)
      );
#endif
      printf("Total number of steal BAD requests  : %"PRIi64 ", %" PRIi64 "\n",
        KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALREQ)-
        KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALREQOK),
        KAAPI_PERF_REG_USR(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALREQ)-
        KAAPI_PERF_REG_USR(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALREQOK)
      );
      printf("Total number of steal operations    : %"PRIi64"\n",
        KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALOP)
      );
      printf("Total number of input steal request : %"PRIi64"\n",
        KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_STEALIN)
      );
      printf("Total number of suspend operations  : %"PRIi64"\n",
        KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_SUSPEND)
      );
      printf("Total number of transfer H2D        : %"PRIi64"\n",
        KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_COMM_OUT)
      );
      printf("Total number of transfer D2H        : %"PRIi64"\n",
        KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_COMM_IN)
      );
      printf("Total compute time                  : %e\n",
         1e-9*(double)KAAPI_PERF_REG_USR(kaapi_all_kprocessors[i], KAAPI_PERF_ID_T1));

      printf("Total compute time of dependencies  : %e\n",
         1e-9*(double)KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], KAAPI_PERF_ID_TASKLISTCALC));

#if defined(KAAPI_USE_PAPI)
      for (int cnt=KAAPI_PERF_ID_PAPI_BASE; cnt<kaapi_mt_perf_counter_num(); ++cnt)
      {
        printf("Papi counter %-14s         : %"PRIi64", %" PRIi64 "\n",
          kaapi_perf_id_to_name( cnt ),
          KAAPI_PERF_REG_USR(kaapi_all_kprocessors[i], cnt),
          KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i], cnt)
        );
      }
#endif

      printf("Total idle time                     : %e\n",
         1e-9*(KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i],KAAPI_PERF_ID_T1)
       + KAAPI_PERF_REG_SYS(kaapi_all_kprocessors[i],KAAPI_PERF_ID_TPREEMPT)) );
    }
  }

  /* */
  if (kaapi_default_param.display_perfcounter)
  {
    printf("----- Cumulated Performance counters\n");
    printf("Total number of tasks executed      : %" PRIu64 "\n", cnt_tasks);
    printf("Total number of steal OK requests   : %" PRIu64 "\n", cnt_stealreqok);
    printf("Total number of steal BAD requests  : %" PRIu64 "\n", cnt_stealreq-cnt_stealreqok);
    printf("Total number of steal operations    : %" PRIu64 "\n", cnt_stealop);
    printf("Total number of input steal request : %" PRIu64 "\n", cnt_stealin);
    printf("Total number of suspend operations  : %" PRIu64 "\n", cnt_suspend);
    printf("Total number of transfers H2D       : %" PRIu64 "\n", cnt_comm_out);
    printf("Total number of transfers D2H       : %" PRIu64 "\n", cnt_comm_in);
    printf("Total compute time                  : %e\n", t_1);
    printf("Total compute time of dependencies  : %e\n", t_tasklist);
    printf("Total idle time                     : %e\n", t_sched+t_preempt);
    printf("   sched idle time                  : %e\n", t_sched);
    printf("   preemption idle time             : %e\n", t_preempt);
    if (cnt_stealop >0)
      printf("Average steal requests aggregation  : %e\n", ((double)cnt_stealreq)/(double)cnt_stealop);
#if defined(KAAPI_USE_PAPI)
    for (int cnt=KAAPI_PERF_ID_PAPI_BASE; cnt<kaapi_mt_perf_counter_num(); ++cnt)
    {
      printf("Papi counter %-14s         : %"PRIi64", %" PRIi64 "\n",
        kaapi_perf_id_to_name( cnt ),
        papicnt[0][cnt-KAAPI_PERF_ID_PAPI_BASE],
        papicnt[1][cnt-KAAPI_PERF_ID_PAPI_BASE]
      );
    }
#endif

  }
#endif
}

