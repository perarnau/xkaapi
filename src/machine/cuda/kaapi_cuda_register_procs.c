/*
 ** kaapi_cuda.c
 ** 
 **
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@imag.fr
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
#include <stdlib.h>
#include <cuda_runtime_api.h>

#include "kaapi_impl.h"
#include "kaapi_cuda.h"
#include "../common/kaapi_procinfo.h"

/* exported */
int kaapi_cuda_register_procs(kaapi_procinfo_list_t* kpl)
{
  const char* const gpuset_str = getenv("KAAPI_GPUSET");
  const char* const gpucount_str = getenv("KAAPI_GPUCOUNT");
  kaapi_procinfo_t* pos = kpl->tail;
  unsigned int kid = kpl->count;
  int devcount;
  int err;
  cudaError_t res;

  if (gpuset_str == NULL)
    return 0;

  if ( (res = cudaGetDeviceCount(&devcount)) != cudaSuccess ) 
  {
    fprintf( stdout, "%s: cudaGetDeviceCount ERROR %d\n",
        __FUNCTION__, res );
    fflush( stdout );
    abort();
  }
  
  if (devcount == 0)
    return 0;
 
  if (gpucount_str != NULL) 
  {
    kaapi_default_param.gpucount = atoi(getenv("KAAPI_GPUCOUNT"));
    if( kaapi_default_param.gpucount > devcount )
      kaapi_default_param.gpucount = devcount;
  } else {
    kaapi_default_param.gpucount = devcount;
  }

  err = kaapi_procinfo_list_parse_string(
	  kpl, gpuset_str, KAAPI_PROC_TYPE_CUDA,
	  kaapi_default_param.gpucount
  );
  if (err) return -1;

  if (kpl->tail == NULL) return 0;

  /* affect kids */
  if (pos == NULL) pos = kpl->tail;
  else pos = pos->next;
  for (; pos; pos = pos->next, ++kid)
    pos->kid = kid;

  return 0;
}


void kaapi_exec_cuda_task
(kaapi_task_t* task, kaapi_thread_t* thread)
{
  kaapi_task_getbody(task)(task->sp, thread);
}
