
#include <stdio.h>
#include <cuda.h>

#include "kaapi_impl.h"
#include "kaapi_tasklist.h"

#include "kaapi_cuda_proc.h"
#include "kaapi_cuda_mem.h"
#include "kaapi_cuda_pool.h"

#if KAAPI_CUDA_USE_POOL

/* cuda task body */
typedef void (*cuda_task_body_t)(void*, cudaStream_t);

static inline void 
kaapi_cuda_pool_new_event( kaapi_cuda_pool_node_t* node )
{
    const cudaError_t res = cuEventCreate( &node->event, CU_EVENT_DISABLE_TIMING );
    if( res != CUDA_SUCCESS ) {
	fprintf( stdout, "[%s] ERROR cuEventCreate %d\n",
		__FUNCTION__, res );
	fflush(stdout);
    }

#if KAAPI_VERBOSE
    fprintf(stdout, "[%s] kid=%lu\n", __FUNCTION__,
		(unsigned long int)kaapi_get_current_kid() );
    fflush( stdout );
#endif
}

static inline kaapi_cuda_pool_node_t* 
kaapi_cuda_pool_node_new( void )
{
    return (kaapi_cuda_pool_node_t*)calloc( 1,
	    sizeof(kaapi_cuda_pool_node_t) );
}

static inline void
kaapi_cuda_pool_node_free( kaapi_cuda_pool_node_t* node )
{
    /* TODO use kaapi_data_t to set as not dirty on GPU */
    cuEventDestroy( node->event );
    free( node );
}

static inline void
kaapi_cuda_pool_launch_kernel( const kaapi_processor_t* proc,
       	kaapi_cuda_pool_node_t* node )
{
#if KAAPI_VERBOSE
    fprintf(stdout, "[%s] kid=%lu\n", __FUNCTION__,
		(unsigned long int)kaapi_get_current_kid() );
    fflush( stdout );
#endif

    if( NULL == proc->cuda_proc.pool->kernel_end ) 
	proc->cuda_proc.pool->kernel_beg= node;
    else
	proc->cuda_proc.pool->kernel_end->next= node;
    proc->cuda_proc.pool->kernel_end= node;
    node->next= NULL;

    cuda_task_body_t body = (cuda_task_body_t)
      node->td->fmt->entrypoint_wh[proc->proc_type];

    body( node->pc->sp, kaapi_cuda_kernel_stream() );

    const cudaError_t res = cuEventRecord( node->event, kaapi_cuda_kernel_stream() );
    if( res != CUDA_SUCCESS ) {
	fprintf( stdout, "[%s] ERROR cuEventCreate (launch) %d\n",
		__FUNCTION__, res );
	fflush(stdout);
    }
}

void
kaapi_cuda_pool_init( kaapi_cuda_proc_t* proc )
{
#if KAAPI_VERBOSE
    fprintf(stdout, "[%s] kid=%lu\n", __FUNCTION__,
		(unsigned long int)kaapi_get_current_kid() );
    fflush( stdout );
#endif
    proc->pool= (kaapi_cuda_pool_t*) calloc( 1, 
	    sizeof(kaapi_cuda_pool_t) );
    proc->pool->size= 0;
    proc->pool->htod_beg= NULL;
    proc->pool->htod_end= NULL;
    proc->pool->kernel_beg= NULL;
    proc->pool->kernel_beg= NULL;
}

int
kaapi_cuda_pool_submit(
	kaapi_thread_context_t* thread,
	kaapi_taskdescr_t*         td,
	kaapi_task_t*              pc
    )
{
    cudaError_t res;
    kaapi_processor_t* const proc = kaapi_get_current_processor();
    kaapi_cuda_pool_node_t* node = kaapi_cuda_pool_node_new();
    node->next = NULL;
    node->td= td;
    node->pc= pc;

#if KAAPI_VERBOSE
    fprintf(stdout, "[%s] kid=%lu\n", __FUNCTION__,
		(unsigned long int)kaapi_get_current_kid() );
    fflush( stdout );
#endif

    kaapi_cuda_mem_sync_params( thread, td, pc );

    /* checkpoint to the kernel execution */
    kaapi_cuda_pool_new_event( node );
    res = cuEventRecord( node->event, kaapi_cuda_HtoD_stream() );
    if( res != CUDA_SUCCESS ) {
	fprintf( stdout, "[%s] ERROR cuEventCreate (HtoD) %d\n",
		__FUNCTION__, res );
	fflush(stdout);
    }

    /* insert new event (HtoD) */
    if( NULL == proc->cuda_proc.pool->htod_end )
	proc->cuda_proc.pool->htod_beg= node;
    else
	proc->cuda_proc.pool->htod_end->next= node;
    proc->cuda_proc.pool->htod_end= node;

    return 0;
}


int
kaapi_cuda_pool_wait( kaapi_thread_context_t* thread )
{
    kaapi_cuda_pool_node_t* node;
    kaapi_processor_t* const proc = kaapi_get_current_processor();

#if KAAPI_VERBOSE
    fprintf(stdout, "[%s] kid=%lu\n", __FUNCTION__,
		(unsigned long int)kaapi_get_current_kid() );
    fflush( stdout );
#endif

    do {
	node= proc->cuda_proc.pool->htod_beg;
        if( ( NULL != node ) &&
    	        ( CUDA_SUCCESS == cuEventQuery(node->event) ) ){
	    proc->cuda_proc.pool->htod_beg= node->next;
	    if( node->next == NULL )
		proc->cuda_proc.pool->htod_end= NULL;
	    kaapi_cuda_pool_launch_kernel( proc, node );
        }

	node= proc->cuda_proc.pool->kernel_beg;
        if( ( NULL != node ) &&
    	        ( CUDA_SUCCESS == cuEventQuery(node->event) ) ){
	    proc->cuda_proc.pool->kernel_beg= node->next;
	    if( node->next == NULL )
		proc->cuda_proc.pool->kernel_end= NULL;
	    kaapi_cuda_mem_sync_params_dtoh( thread, node->td, node->pc );
	    kaapi_cuda_pool_node_free( node );
	}

    } while( (NULL != proc->cuda_proc.pool->htod_beg) || 
	     (NULL != proc->cuda_proc.pool->kernel_beg) );

    return 0;
}

#endif /* KAAPI_CUDA_USE_POOL */
