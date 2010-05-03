/*
 ** kaapi_task_checkdenpendencies.c
 ** xkaapi
 ** 
 ** Created on Tue Feb 23 16:56:43 2010
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 **
 ** theo.trouillon@imag.fr
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


/** Global Hashmap for WS
 */
kaapi_hashmap_t ws_khm;

/**
 */
void kaapi_task_checkdependencies(kaapi_thread_t* thread)
{
  kaapi_task_t* task=(kaapi_task_t*)thread->sp;
  kaapi_format_t* format;
  if(task->body==kaapi_suspend_body || task->body==kaapi_exec_body)
    format= kaapi_format_resolvebybody(task->ebody);
  else 
    format= kaapi_format_resolvebybody(task->body);
  
  if (format!=0 && format!=NULL)
  {
    kaapi_hashentries_t* entry; //Current argument's entry in the Hashmap
    kaapi_atomic_t* counter=0; //Writers tasks counter
    
    for (int i=0;i<format->count_params;i++) //Loop on each argument of the task
    {
      //printf("[CHECKDEPS]%d:::%d\n",i,task->sp+format->off_params[i]);
      entry=kaapi_hashmap_find(&ws_khm,task->sp+format->off_params[i]);
      
      if(entry!=NULL && (KAAPI_ACCESS_IS_READ(format->mode_params[i]) || KAAPI_ACCESS_IS_READWRITE(format->mode_params[i])) && ! ( ( thread <= ( entry->datas->last_writer) ) && (( entry->datas->last_writer)<= (thread + (kaapi_default_param.stacksize) ) ) ) )
        //Argument is already referenced (previous writer exist), and this task will r/rw current argument, and last writer task is NOT in the same stack
      {
        if ((entry->datas->last_writer)>(entry->datas->last_writer_thread->pc))
        {
          goto already_terminated;
        }	        
        kaapi_task_setextrabody(task,task->body);	
        kaapi_task_setbody( task, kaapi_suspend_body);
        
        
        if (counter==0)//task counter creation
        {
          counter=(kaapi_atomic_t*)kaapi_thread_pushdata_align(thread, sizeof(kaapi_atomic_t),8);
          KAAPI_ATOMIC_WRITE(counter,1);
        }
        else //task counter update
        {
          KAAPI_ATOMIC_INCR(counter);
        }
        
        //Signal datas creation
        
        kaapi_counters_list* new_reader_a= (kaapi_counters_list*)kaapi_thread_pushdata(entry->datas->last_writer_thread, sizeof(kaapi_counters_list));
        new_reader_a->next=0;
        new_reader_a->reader_counter=counter;
        new_reader_a->waiting_task=task;

	kaapi_readmem_barrier();
        if (entry->datas->last_writer->pad!=0)
          //last_writer has already dependency datas, update its datas
        {
          kaapi_counters_list* tmp=(kaapi_counters_list*)(entry->datas->last_writer->pad);
          //printf("task:%u;last:%u;counter:%u,counter_addr:%u",task,(kaapi_deps_t*)(entry->datas)->last_writer,signal_datas->readers_number._counter, &(signal_datas->readers_number)); 
          while(tmp->next != 0) //may be locked?
          {
            tmp=tmp->next;
          }
          tmp->next=new_reader_a;
          //printf("Datas updated\n");
        }
        else //Set datas
        {
          entry->datas->last_writer->pad=(void*)new_reader_a; 
          //printf("task:%u;last:%u;counter:%u,counter_addr:%u",task,(kaapi_deps_t*)(entry->datas)->last_writer,signal_datas->readers_number._counter, &(signal_datas->readers_number)); 
          //printf("Datas set\n");
        }

	//if writer terminated while datas passing, test if he saw it, if not, correct the counter
	// !!! Incorrect issues are possible (no decrementation of the counter, task will keep suspended state). TODO
	if ((entry->datas->last_writer)>(entry->datas->last_writer_thread->pc))
        {
		kaapi_readmem_barrier();	
		if(entry->datas->last_writer->pad!=0)
			KAAPI_ATOMIC_DECR(counter);
        }
      }
	  already_terminated:
      if(KAAPI_ACCESS_IS_WRITE(format->mode_params[i]) || KAAPI_ACCESS_IS_READWRITE(format->mode_params[i]))//This task will w/rw current argument
      {
        if (entry==NULL) //argument not referenced
        {
          entry=kaapi_hashmap_insert(&ws_khm,task->sp+format->off_params[i]);
        }
        //Update argument's last writer informations
        entry->datas->last_writer=task;
        entry->datas->last_writer_thread=thread;
	      //printf("Last writer inserted\n");
      }
    }
  }
  
}

/**
 */
/* No more body replacement
 
 void kaapi_dependenciessignal_body( kaapi_task_t* task, kaapi_stack_t* stack )
 {
 printf("Signal body:%u;stack:%u\n",task,stack);
 kaapi_dependenciessignal_arg_t* signal_data = kaapi_task_getargst( task, kaapi_dependenciessignal_arg_t); 
 //Recovery
 task->sp=signal_data->real_datas;
 task->format=signal_data->real_format;
 task->splitter=signal_data->real_splitter;
 
 
 (*(signal_data->real_body))(task,stack); //Execution of the real body
 
 //Reset the task flag to KAAPI_TASK_S_TERM   
 //task->flag&=0xFFFFFF0F;
 //task->flag|=KAAPI_TASK_S_TERM;
 
 kaapi_atomic_t* counter;
 
 while(signal_data->readers_list != NULL) //counters decrementation, if 0: wake up
 {
 counter=signal_data->readers_list->reader_counter;
 KAAPI_ATOMIC_DECR(counter);
 if ((counter->_counter)==0)
 {
 //printf("[CHECK-SB:Waking up");
 //kaapi_task_setstate(signal_data->readers_list->waiting_task,signal_data->readers_list->origin_state);
 signal_data->readers_list->waiting_task->flag&=0xFFFFFF0F;
 }
 signal_data->readers_list=signal_data->readers_list->next;
 }
 
 //printf("Signal body terminated\n");
 }
 */
