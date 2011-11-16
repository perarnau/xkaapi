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

#if defined(KAAPI_USE_CUDA)

static void kaapi_cuda_taskmove_body(void* sp, kaapi_thread_t* thread)
{
  kaapi_move_arg_t* const arg = (kaapi_move_arg_t*)sp;

  kaapi_processor_t* const proc = kaapi_get_current_processor();

  printf("%s: [%u:%u] (%lx:%lx -> %lx:%lx) %lx\n",
	 __FUNCTION__,
	 proc->kid, proc->proc_type,
	 arg->src_data->ptr.asid, (uintptr_t)arg->src_data->ptr.ptr,
	 arg->dest->ptr.asid, (uintptr_t)arg->dest->ptr.ptr,
	 (uintptr_t)arg->dest->mdi);

}

static void kaapi_cuda_taskalloc_body(void* sp, kaapi_thread_t* thread)
{
  kaapi_move_arg_t* const arg = (kaapi_move_arg_t*)sp;

  kaapi_processor_t* const proc = kaapi_get_current_processor();

  printf("%s: [%u:%u] (%lx:%lx -> %lx:%lx) %lx\n",
	 __FUNCTION__,
	 proc->kid, proc->proc_type,
	 arg->src_data->ptr.asid, (uintptr_t)arg->src_data->ptr.ptr,
	 arg->dest->ptr.asid, (uintptr_t)arg->dest->ptr.ptr,
	 (uintptr_t)arg->dest->mdi);

}

#endif /* KAAPI_USE_CUDA */

/* */
void kaapi_taskmove_body( void* sp, kaapi_thread_t* thread)
{
  kaapi_move_arg_t* arg = (kaapi_move_arg_t*)sp;

#if defined(KAAPI_USE_CUDA)
  /* todo: kaapi_cuda_taskmove_body should be directly pushed instead */
  if (kaapi_get_current_processor()->proc_type == KAAPI_PROC_TYPE_CUDA)
  {
    kaapi_cuda_taskmove_body(sp, thread);
    return ;
  }
#endif

  /* on multiprocessor: move data from XXX to YYY */
  arg->dest->ptr  = arg->src_data.ptr;
}

/* */
void kaapi_taskalloc_body( void* sp, kaapi_thread_t* thread )
{
  kaapi_move_arg_t* arg = (kaapi_move_arg_t*)sp;

#if defined(KAAPI_USE_CUDA)
  /* todo: kaapi_cuda_taskalloc_body should be directly pushed instead */
  if (kaapi_get_current_processor()->proc_type == KAAPI_PROC_TYPE_CUDA)
  {
    kaapi_cuda_taskalloc_body(sp, thread);
    return ;
  }
#endif

  /* on multiprocessor: move data from XXX to YYY */
  arg->dest->ptr  = arg->src_data.ptr;
}


/* */
void kaapi_taskfinalizer_body( void* sp, kaapi_thread_t* thread )
{
//  kaapi_move_arg_t* arg = (kaapi_move_arg_t*)sp;
}

