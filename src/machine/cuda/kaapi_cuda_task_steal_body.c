
#include <stdio.h>

#include <cuda_runtime_api.h>

#include "kaapi_impl.h"

/* cuda task body */
typedef void (*cuda_task_body_t) (void *, cudaStream_t);

int kaapi_cuda_task_steal_body(kaapi_thread_t * thread,
			   const kaapi_format_t * fmt, void *sp)
{
  fprintf(stdout, "%s:%d:%s: ERROR not supported.\n",
          __FILE__, __LINE__, __FUNCTION__
          );
  fflush(stdout);
  kaapi_abort();
  return 0;
}
