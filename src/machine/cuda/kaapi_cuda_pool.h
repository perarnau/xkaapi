
#ifndef KAAPI_CUDA_POOL_H
#define KAAPI_CUDA_POOL_H

#include "kaapi.h"
#include "kaapi_cuda_proc.h"

typedef struct kaapi_cuda_pool_node {
    CUevent event;
    kaapi_taskdescr_t*         td;
    kaapi_task_t*              pc;
    struct kaapi_cuda_pool_node* next;
} kaapi_cuda_pool_node_t;

typedef struct kaapi_cuda_pool {
    unsigned int size;
    kaapi_cuda_pool_node_t* htod_beg;
    kaapi_cuda_pool_node_t* htod_end;
    kaapi_cuda_pool_node_t* kernel_beg;
    kaapi_cuda_pool_node_t* kernel_end;
} kaapi_cuda_pool_t;

void kaapi_cuda_pool_init( kaapi_cuda_proc_t* proc );
#if 0
void kaapi_cuda_pool_submit_HtoD( );
void kaapi_cuda_pool_submit_DtoH( );
void kaapi_cuda_pool_submit_kernel( );
#endif

int kaapi_cuda_pool_submit(
	kaapi_thread_context_t* thread,
	kaapi_taskdescr_t*         td,
	kaapi_task_t*              pc
       	);

int kaapi_cuda_pool_wait( kaapi_thread_context_t* thread );

#endif /* KAAPI_CUDA_POOL_H */
