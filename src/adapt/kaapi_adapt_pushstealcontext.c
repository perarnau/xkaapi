/*
** kaapi_task_pushstealcontext.c
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

/* good name ! 
   Wrapper between the old splitter interface and the new one.
*/
static int kaapi_adaptivetask_splitter_2_usersplitter( 
    kaapi_task_t*                 pc, /* this is the adaptive task on which splitter is called */
    void*                         sp,
    kaapi_listrequest_t*          lrequests, 
    kaapi_listrequest_iterator_t* lrrange
)
{
  /* - pc is kaapi_taskadapt_body 
  */
  kaapi_taskbegendadaptive_arg_t* arg = (kaapi_taskbegendadaptive_arg_t*)sp;
  kaapi_assert_debug( pc !=0 );

  /* call the splitter */
  return arg->usersplitter( 
      pc,
      arg->argsplitter,
      lrequests,
      lrrange
  );
}

#if defined(KAAPI_DEBUG)
static int volatile global_version = 0;
#endif


/* Ancienne interface basee sur la nouvelle api des taches adaptatives.
   - la creation d'une tache adaptative cree 2 taches
      * la premiere est la tache qui sera decoupe par un splitter
      * la seconde sert de synchronisation de fin d'execution de la premiere (toujours garantis...)
      * les deux taches ont une dependance sur le steal context (rw) pour les deux taches
      * des dependances sur les donnes de la tache adaptative utilisateur pourraient etre aussi definis
   - tous les vols sur 
*/
void* kaapi_task_begin_adaptive
(
  kaapi_thread_t*               thread,
  int                           flag,
  kaapi_adaptivetask_splitter_t splitter,
  void*                         argsplitter
)
{
  kaapi_stealcontext_t*   sc;
  kaapi_task_t*           task_adapt;
    
  /* */
  kaapi_taskbegendadaptive_arg_t* adap_arg;

#if 0 //OLD is user spawns tasks, then he should call sync
  /* new frame for all tasks between begin..end */
  kaapi_thread_push_frame();
#endif
      
  /* allocated tasks' args */
  adap_arg = (kaapi_taskbegendadaptive_arg_t*)kaapi_thread_pushdata
    (thread, sizeof(kaapi_taskbegendadaptive_arg_t));
  kaapi_assert_debug(adap_arg != 0);

  sc = (kaapi_stealcontext_t*)kaapi_thread_pushdata_align
    (thread, sizeof(kaapi_stealcontext_t), sizeof(void*));
  kaapi_assert_debug(sc != 0);

  kaapi_access_init(&adap_arg->shared_sc, sc);
  adap_arg->splitter    = kaapi_adaptivetask_splitter_2_usersplitter;
  adap_arg->usersplitter= splitter;
  adap_arg->argsplitter = argsplitter;
  sc->msc               = sc; /* self pointer to detect master */
  sc->ktr               = 0;
  
  if (flag & KAAPI_SC_PREEMPTION)
  {
    kaapi_assert_debug( !(flag & KAAPI_SC_NOPREEMPTION) );
    /* if preemption, thief list used ... */
    KAAPI_ATOMIC_WRITE(&sc->thieves.list.lock, 0);
    sc->thieves.list.head = 0;
    sc->thieves.list.tail = 0;
  }
  else /* no preemption */
  {
    kaapi_assert_debug( !(flag & KAAPI_SC_PREEMPTION) );
    /* ... otherwise thief count */
    KAAPI_ATOMIC_WRITE(&sc->thieves.count, 0);
  }

#if defined(KAAPI_DEBUG)
  sc->version = ++global_version; // assume only main thread incr global_version
  sc->state = 1;
#endif

/*
  ICI mettre le code de 'split before publishing the task'
*/
  if (flag & KAAPI_SC_INITIALSPLIT)
  {
    /* todo: levelid should be an argument */
//    static const kaapi_hws_levelid_t levelid = KAAPI_HWS_LEVELID_NUMA;
//    kaapi_hws_splitter(sc, splitter, argsplitter, levelid);
  }

  /* create the adaptive task:
     - it is a dummy task that represents the current executing 
     bloc between begin/end_adaptive
     - it is splittable but not stealable
  */
  task_adapt = kaapi_thread_toptask(thread);
  kaapi_task_init_with_flag(
      task_adapt, 
      (kaapi_task_body_t)kaapi_taskbegendadapt_body, 
      adap_arg, 
      KAAPI_TASK_UNSTEALABLE| KAAPI_TASK_SPLITTABLE | flag
  );
  kaapi_task_setstate( task_adapt, KAAPI_TASK_STATE_EXEC );

  /* memory barrier done by kaapi_thread_pushtask */
  kaapi_thread_pushtask(thread);

  return task_adapt;
}
