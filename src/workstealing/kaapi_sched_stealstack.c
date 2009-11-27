/*
** kaapi_sched_stealstack.c
** xkaapi
** 
** Created on Tue Mar 31 15:18:04 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
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

/** 
L'algorithme actuel est un peu trop simple et ne marche que pour
des tâches qui serait dans une même frame.
Dans le cas de programme récursif, l'algorithme devrait être celui-ci:

- 1/ identification de la frame courante:
  - identification de la prochaine tâche retn
  - les tâches après frame->save_sp et une autre tâche retn constitue
  la frame de la tâche frame->save_pc.
  Soit (ibeg,iend) les tâches: 1ère tâche forkée de la frame (et peut-être exécuté)
  et iend la tâche retn qui marquent la frame.
  
- 2/ calcul des versions à lire:
Input: (ibeg, iend), pc la tâche qui s'exécute ou va s'exécuter
Output: version des gd de chacune des tâches qui pointe sur la version à lire (R ou RW) ou 0
si l'accès n'est pas prêt.

  - forall t in (ibeg,iend) trois cas:
      (a) si t < pc: t est déjà exécutée ainsi que toute sa descendence. 
      (b) si t == pc: 
        (b.1) si sp > iend alors d'autres tâches ont été forkées
        (b.2) sinon -> t est "en cours d'exécution".
      (c) si t > pc et t < iend : les tâches ne sont pas encores exécutées.
  - cas (a): forall gd in t => last_version : gd->data
  - cas (b.1): 
      - calcul des version de la frame (iend+1, ... next retn).
  - cas (b.2):
      - forall gd in t, mode(gd) = RW ou W ou CW alors last version =0
  - cas (c): idem algo actuel

- 3/ calcul d'une tâche à voler:
  - parcours de la liste des tâches / ordre de la pile, prendre la première prête

Remarque: il faudra bien identifier une tâche en cours d'exéc ou une tâche terminée.
les cas (b.1) et (b.2) devront être unifié : en cours d'exécution == toute la descendance non encore
exécutée.
  - structure tâche:
      - body (void*) -> 32 ou 64 bits
      - sp (void*) -> 32 / 64 bits ou alors offset 32 bits ou moins ???? si offset / sp_data debut de frame ou data de la pile
      - flags: 4 bits atributs, 3 bits processor types, -> 25 bits free !
        - format: local number : < 256 -> 8bits + table
        - attributs: 4 bits
        - processor type: 4 bits
        - state: 2 bits: Init (0) -> Exec(1) -> Steal (2) -> Term (3).
      
*/
int kaapi_sched_stealstack  ( kaapi_stack_t* stack )
{
  kaapi_task_t*  task_top;
  kaapi_task_t*  task_bot;
  int count;
  int replycount;  

  count = KAAPI_ATOMIC_READ( (kaapi_atomic_t*)stack->hasrequest );
  if (count ==0) return 0;

  if (kaapi_stack_isempty( stack)) return 0;

printf("------ STEAL STACK @:%p\n", (void*)stack );
  /* reset dfg constraints evaluation */
  
  /* iterate through all the tasks from task_bot until task_top */
  task_bot = kaapi_stack_bottomtask(stack);
  task_top = kaapi_stack_toptask(stack);

  replycount = 0;

  while ((count >0) && (task_bot !=0) && (task_bot != task_top))
  {
    if (task_bot == 0) {
printf("------ END STEAL @:%p\n", (void*)stack );
      return replycount;
    }

    kaapi_assert_debug( task_bot != 0 );
    /* task body == 0 no task after can stop 
       task body == retn : no steal
       
    */
    if (task_bot->splitter !=0)
    {
      int retval = (*task_bot->splitter)(stack, task_bot, count, stack->requests);
      count -= retval;
      replycount += retval;
      kaapi_assert_debug( count >=0 );
    }

    /* test next task */  
    ++task_bot;
  }
printf("------ END STEAL @:%p\n", (void*)stack );
  
  if (replycount >0)
  {
    KAAPI_ATOMIC_SUB( (kaapi_atomic_t*)stack->hasrequest, replycount );
    kaapi_assert_debug( *stack->hasrequest >= 0 );
  }

  return replycount;
}
