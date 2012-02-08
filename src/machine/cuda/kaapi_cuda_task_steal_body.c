
#include <stdio.h>

#include "kaapi_impl.h"
#include "kaapi_cuda_proc.h"
#include "kaapi_cuda_ctx.h"
#include "kaapi_cuda_data.h"
#include "kaapi_cuda_task_steal_body.h"
#include "kaapi_cuda_execframe.h"

/* cuda task body */
typedef void (*cuda_task_body_t)(void*, cudaStream_t);

int
kaapi_cuda_task_steal_body(
	kaapi_thread_t*	thread,
	kaapi_format_t* fmt,
	void* sp
	)
{
    cudaError_t res;
    cuda_task_body_t body = (cuda_task_body_t)
	fmt->entrypoint_wh[KAAPI_PROC_TYPE_CUDA];
    /* Enter CUDA context */
    kaapi_cuda_ctx_push( );

    kaapi_cuda_data_send_ptr( fmt, sp );
    res = cuCtxSynchronize( );
    if( res != cudaSuccess ) {
	    fprintf( stdout, "[%s] CUDA kernel ERROR: %d\n", __FUNCTION__, res);
	    fflush(stdout);
    }
    body( sp, kaapi_cuda_kernel_stream() );
    res = cuCtxSynchronize( );
    if( res != cudaSuccess ) {
	    fprintf( stdout, "[%s] CUDA kernel ERROR: %d\n", __FUNCTION__, res);
	    fflush(stdout);
    }
    kaapi_cuda_data_recv_ptr( fmt, sp );
    res = cuCtxSynchronize( );
    if( res != cudaSuccess ) {
	    fprintf( stdout, "[%s] CUDA kernel ERROR: %d\n", __FUNCTION__, res);
	    fflush(stdout);
    }
    /* Exit CUDA context */
    kaapi_cuda_ctx_pop( );

    return res;
}

#if 0
{
    int res;
    const size_t count_params = kaapi_format_get_count_params(fmt, sp );
    size_t i;
    kaapi_version_t*        version;
    int                     islocal;
    kaapi_taskdescr_t*	    td;
    kaapi_tasklist_t	    tl;
    kaapi_handle_t	    handle;
    kaapi_thread_context_t* const self_thread = kaapi_self_thread_context();

    res = kaapi_tasklist_init(&tl);

    for ( i=0; i < count_params; i++ ) {
	    kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(
		    kaapi_format_get_mode_param( fmt, i, sp) );
	    if (m == KAAPI_ACCESS_MODE_V) 
		    continue;

	    kaapi_assert_debug( m != KAAPI_ACCESS_MODE_VOID );
	    kaapi_access_t access = kaapi_format_get_access_param( fmt,
			    i, sp );

	    version = kaapi_version_findinsert( &islocal, thread, tasklist, access.data );
	    if (version->last_mode == KAAPI_ACCESS_MODE_VOID)
	    {
	      kaapi_memory_view_t view = kaapi_format_get_view_param(task_fmt, i, task->sp);
	      kaapi_version_add_initialaccess( version, tasklist, m, access.data, &view );
	      islocal = 1;
	    }
	    handle = kaapi_thread_computeready_access( tasklist, version, taskdescr, m );
	    /* replace the pointer to the data in the task argument by the pointer to the global data */
	    access.data = handle;
	    kaapi_format_set_access_param(task_fmt, i, task->sp, &access);
	    
	    kaapi_data_t* src = kaapi_data( kaapi_data_t, &access );
	    kaapi_mem_host_map_find_or_insert( host_map,
		    (kaapi_mem_addr_t)kaapi_pointer2void(src->ptr),
		    &kmd );
	    if( !kaapi_mem_data_has_addr( kmd, host_asid ) )
		kaapi_mem_data_set_addr( kmd, host_asid,
			(kaapi_mem_addr_t)src );

	    kaapi_data_t* dest = xxx_kaapi_cuda_data_allocate( cuda_map, kmd, src );

	    if( KAAPI_ACCESS_IS_READ(m) )
		    xxx_kaapi_cuda_data_send_ro( cuda_map, kmd, dest, src );

	    if( KAAPI_ACCESS_IS_WRITE(m) )
		    xxx_kaapi_cuda_data_send_wr( cuda_map, kmd, dest, src );

	    /* sets new pointer to the task */
	    access.data =  dest;
	    kaapi_format_set_access_param( fmt, i, sp, &access );
    }
    return 0;
}
#endif

#if 0
{
    cudaError_t res;
    cuda_task_body_t body = (cuda_task_body_t)
	fmt->entrypoint_wh[KAAPI_PROC_TYPE_CUDA];

    /* Enter CUDA context */
    kaapi_cuda_ctx_push( );
    kaapi_cuda_data_send( fmt, sp );
    res = cuCtxSynchronize( );
    if( res != cudaSuccess ) {
	fprintf( stdout, "[%s] CUDA kernel ERROR: %d\n", __FUNCTION__, res);
	fflush(stdout);
    }
    body( sp, kaapi_cuda_kernel_stream() );
    res = cuCtxSynchronize( );
    if( res != cudaSuccess ) {
	fprintf( stdout, "[%s] CUDA kernel ERROR: %d\n", __FUNCTION__, res);
	fflush(stdout);
    }
    kaapi_cuda_data_recv( fmt, sp );
    res = cuCtxSynchronize( );
    cuStreamSynchronize( kaapi_cuda_DtoH_stream() );
    if( res != cudaSuccess ) {
	fprintf( stdout, "[%s] CUDA kernel ERROR: %d\n", __FUNCTION__, res);
	fflush(stdout);
    }
    /* Exit CUDA context */
    kaapi_cuda_ctx_pop( );

    return res;
}
#endif
